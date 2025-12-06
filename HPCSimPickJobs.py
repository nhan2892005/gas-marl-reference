from job import Job, Workloads
from cluster import Cluster
import sys
import copy
import numpy as np
import scipy.signal
import gym
from gym import spaces
from gym.utils import seeding
import configparser
from sentence_transformers import SentenceTransformer
import torch

config = configparser.ConfigParser()
config.read('configFile/config.ini')


eta = float(config.get('GAS-MARL setting', 'eta'))
MAX_QUEUE_SIZE = int(config.get('GAS-MARL setting', 'MAX_QUEUE_SIZE'))
run_win = int(config.get('GAS-MARL setting', 'run_win'))
green_win = int(config.get('GAS-MARL setting', 'green_win'))
delayMaxJobNum = int(config.get('GAS-MARL setting', 'delayMaxJobNum'))
delayTimeList = eval(config.get('GAS-MARL setting', 'delayTimeList'))
embbedVectorNum = int(config.get('GAS-MARL setting', 'embbedVectorNum'))
embbedVectorSize = int(config.get('GAS-MARL setting', 'embbedVectorSize'))

turbinePowerNominal = float(config.get('general setting', 'turbinePowerNominal'))
numberPv = float(config.get('general setting', 'numberPv'))
processor_per_machine = int(config.get('general setting', 'processor_per_machine'))
idlePower = float(config.get('general setting', 'idlePower'))
MAX_perProcPower = float(config.get('general setting', 'MAX_perProcPower'))


MAX_POWER = 19000
MAX_GREEN = 19000
MAX_WAIT_TIME = 12 * 60 * 60
MAX_RUN_TIME = 12 * 60 * 60
JOB_FEATURES = 8
JOB_SEQUENCE_SIZE = MAX_QUEUE_SIZE
RUN_FEATURE = 4
GREEN_FEATURE = 2


action2_num= len(delayTimeList) + delayMaxJobNum + 1

model_name = "sentence-transformers/all-MiniLM-L6-v2"

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class HPCEnv(gym.Env):
    def __init__(self, backfill=False):  # do nothing and return. A workaround for passing parameters to the environment
        super(HPCEnv, self).__init__()
        print("Initialize Simple HPC Env")

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(JOB_FEATURES * MAX_QUEUE_SIZE,),
                                            dtype=np.float32)

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.start_idx_last_reset = 0

        self.loads = None
        self.cluster = None

        self.scheduled_rl = {}

        self.backfill = backfill

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_encoder = SentenceTransformer(model_name, device=device)

    # @profile
    def my_init(self, workload_file='', sched_file=''):
        print("loading workloads from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster("Cluster", self.loads.max_nodes, self.loads.max_procs / self.loads.max_nodes,processor_per_machine,idlePower,green_win,numberPv,turbinePowerNominal)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def f2_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f2: r^(1/2)*n + 25600 * log10(s)
        return (np.sqrt(request_time) * request_processors + 25600 * np.log10(submit_time))


    def fcfs_score(self, job):
        submit_time = job.submit_time
        return submit_time

    def lptpn_score(self, job):
        t = -job.power*job.request_time
        return t

    def backfill_score(self, job):
        t = job.power*job.request_time*job.request_number_of_processors  # Green-Backfilling

        return t

    # @profile
    def reset(self,repre="feature"):
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}

        job_sequence_size = JOB_SEQUENCE_SIZE

        self.start = self.np_random.integers(job_sequence_size, (self.loads.size() - job_sequence_size - 1))

        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

        return self.build_observation(repre=repre)

    def reset_for_test(self, num, start):
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}

        job_sequence_size = num
        self.start = start

        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

    def skip_for_resources_greedy(self, job, scheduled_logs):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        while not self.cluster.can_allocated(job):
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[
                self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    def skip_for_resources_LPTPN(self, job, scheduled_logs):
        # note that this function is only called when current job can not be scheduled.
        # assert not self.cluster.can_allocated(job)

        while not self.cluster.can_allocated(job) or not self.cluster.LPTPN_check(self.running_jobs, job, self.current_timestamp) :
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            # assert self.running_jobs
            if len(self.running_jobs)==0:
                break
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            nextGreenChange = self.current_timestamp + 1
            # nextGreenChange = self.current_timestamp + 1
            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= min(next_resource_release_time,nextGreenChange):
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            elif nextGreenChange < next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, nextGreenChange)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job

    # @profile
    def moveforward_for_resources_backfill_greedy(self, job, scheduled_logs):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        for running_job in self.running_jobs:
            free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if free_processors >= job.request_number_of_processors:
                break

        while not self.cluster.can_allocated(job):

            # try to backfill as many jobs as possible.
            if self.backfill == 1:
                self.job_queue.sort(key=lambda _j: self.backfill_score(_j))
            else:
                self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
            job_queue_iter_copy = list(self.job_queue)

            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
            free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
            temp_est=earliest_start_time
            for running_job in self.running_jobs:
                free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
                temp_est = (running_job.scheduled_time + running_job.request_time)
                if free_processors >= job.request_number_of_processors:
                    break
            earliest_start_time=max(earliest_start_time,temp_est)
            for _j in job_queue_iter_copy:
                if _j!=job and (self.current_timestamp + _j.request_time) < earliest_start_time:
                    if self.cluster.backfill_check(self.running_jobs, _j, self.current_timestamp, self.backfill):
                        # we should be OK to schedule the job now
                        assert _j.scheduled_time == -1  # this job should never be scheduled before.
                        _j.scheduled_time = self.current_timestamp
                        _j.allocated_machines = self.cluster.allocate(_j.job_id, _j.request_number_of_processors)
                        self.cluster.PowerStruc.update(_j.scheduled_time,
                                                       _j.scheduled_time + _j.run_time,
                                                       _j.power)
                        self.running_jobs.append(_j)
                        score = self.job_score(_j)  # calculated reward
                        scheduled_logs[_j.job_id] = score
                        self.job_queue.remove(_j)  # remove the job from job queue

            # move to the next timestamp
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            nextGreenChange = ((self.current_timestamp // 3600) + 1) * 3600
            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= min(next_resource_release_time,nextGreenChange):
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            elif nextGreenChange < next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, nextGreenChange)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job

        self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))

    def post_process_score(self, scheduled_logs):
        for i in scheduled_logs:
            scheduled_logs[i] /= self.num_job_in_batch

    # @profile
    def schedule_curr_sequence_reset(self, score_fn):
        # schedule the sequence of jobs using heuristic algorithm.
        scheduled_logs = {}
        while True:
            self.job_queue.sort(key=lambda j: score_fn(j))
            job_for_scheduling = self.job_queue[0]

            # if selected job needs more resources, skip scheduling and try again after adding new jobs or releasing some resources
            if not self.cluster.can_allocated(job_for_scheduling):
                if self.backfill:
                    self.moveforward_for_resources_backfill_greedy(job_for_scheduling, scheduled_logs)
                else:
                    self.skip_for_resources_greedy(job_for_scheduling, scheduled_logs)

            assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
            job_for_scheduling.scheduled_time = self.current_timestamp
            job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                          job_for_scheduling.request_number_of_processors)
            self.cluster.PowerStruc.update(job_for_scheduling.scheduled_time,
                                           job_for_scheduling.scheduled_time + job_for_scheduling.run_time,
                                           job_for_scheduling.power)
            self.running_jobs.append(job_for_scheduling)
            score = self.job_score(job_for_scheduling)  # calculated reward
            scheduled_logs[job_for_scheduling.job_id] = score
            self.job_queue.remove(job_for_scheduling)

            not_empty = self.moveforward_for_job()
            if not not_empty:
                break
        self.post_process_score(scheduled_logs)
        greenRwd = self.cluster.greenPower.getGreenPowerUtilization(self.cluster.PowerStruc.powerSlotLog)

        self.cluster.reset()
        self.loads.reset()
        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        return scheduled_logs,greenRwd

    def build_observation(self, repre="feature"):
        currentSlot=self.cluster.PowerStruc.getSlotFromRunning(self.running_jobs,self.current_timestamp)
        self.pairs = [
                            [
                                job,
                                min(float(self.current_timestamp - job.submit_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5),
                                min(float(job.request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5),
                                min(float(job.request_number_of_processors) / float(self.loads.max_procs), 1.0 - 1e-5),
                                min(float(job.power) / float(MAX_POWER), 1.0 - 1e-5),
                                min(float(job.power / job.request_number_of_processors) / float(MAX_perProcPower),
                                    1.0 - 1e-5),
                                *self.cluster.getGreenJobState(job, self.current_timestamp, currentSlot),
                                1.0 - 1e-5 if self.cluster.can_allocated(job) else 1e-5
                            ]
                            for i, job in enumerate(self.job_queue)
                            if i < MAX_QUEUE_SIZE
                        ] + [
                            [None, 0, 1, 1, 1, 1, 1, 1, 0]
                            for _ in range(MAX_QUEUE_SIZE - len(self.job_queue))
                        ]
        if repre == "feature":
            vector = np.zeros((MAX_QUEUE_SIZE + run_win + green_win) * JOB_FEATURES, dtype=float)
            self.job_queue.sort(key=lambda job: self.fcfs_score(job))

            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))

            vector[:MAX_QUEUE_SIZE * JOB_FEATURES] = [item for pair in self.pairs[:MAX_QUEUE_SIZE] for item in pair[1:]]

            running_job = [
                            [
                                min(float(temp_job.request_number_of_processors) / float(self.loads.max_procs),
                                    1.0 - 1e-5),
                                min(float(temp_job.power) / float(MAX_POWER), 1.0 - 1e-5),
                                min(float(temp_job.power / temp_job.request_number_of_processors) / float(
                                    MAX_perProcPower), 1.0 - 1e-5),
                                min(float(temp_job.scheduled_time + temp_job.request_time - self.current_timestamp) / float(
                                    self.loads.max_exec_time), 1.0 - 1e-5),
                                0,0,0,0
                            ]
                            for i, temp_job in enumerate(self.running_jobs[:run_win])
                            if i < run_win
                        ] + [
                            [0, 0, 0, 0, 0,0,0,0]
                            for _ in range(run_win - len(self.running_jobs))
                        ]
            vector[MAX_QUEUE_SIZE * JOB_FEATURES:(MAX_QUEUE_SIZE + run_win) * JOB_FEATURES] = [
                job_feature for job in running_job for job_feature in job
            ]

            green = self.cluster.greenPower.getGreenPowerSlot(self.current_timestamp)
            green_slot = [
                [
                    min(float(greenPower['lastTime']) / float(self.loads.max_exec_time), 1.0 - 1e-5),
                    min(float(greenPower['power']) / float(MAX_GREEN), 1.0 - 1e-5),
                    0,0,0,0,0,0
                ]
                for greenPower in green
            ]

            start_index = MAX_QUEUE_SIZE + run_win
            end_index = MAX_QUEUE_SIZE + run_win + green_win
            vector[start_index * JOB_FEATURES:end_index * JOB_FEATURES] = [item for slot in green_slot[
                                                                                            start_index - MAX_QUEUE_SIZE - run_win:end_index - MAX_QUEUE_SIZE - run_win]
                                                                        for item in slot]

            return vector
        elif repre == "text":
            # Tính toán metrics tổng quan của hệ thống
            total_cpus = self.cluster.total_node * self.cluster.num_procs_per_node
            free_cpus = self.cluster.free_node * self.cluster.num_procs_per_node
            used_cpus = total_cpus - free_cpus
            cpu_utilization = (used_cpus / max(1, total_cpus)) * 100
            
            # Giả định power metrics (nếu có trong cluster)
            # Nếu cluster không có thuộc tính này, có thể tính toán từ running jobs
            total_power_capacity = sum(j.power for j in self.running_jobs) if self.running_jobs else 10000
            used_power = sum(j.power for j in self.running_jobs)
            available_power = total_power_capacity - used_power
            
            # --------------------------------------------------------------------------
            # PART 1: JOB QUEUE CONTEXT
            # --------------------------------------------------------------------------
            total_waiting = len(self.job_queue)
            
            if total_waiting > 0:
                avg_wait = sum(self.current_timestamp - j.submit_time for j in self.job_queue) / total_waiting
                total_cpu_demand = sum(j.request_number_of_processors for j in self.job_queue)
                total_power_demand = sum(j.power for j in self.job_queue)
                max_wait = max(self.current_timestamp - j.submit_time for j in self.job_queue)
                min_wait = min(self.current_timestamp - j.submit_time for j in self.job_queue)
            else:
                avg_wait = total_cpu_demand = total_power_demand = max_wait = min_wait = 0
            
            # Câu mở đầu với ngữ cảnh tổng quan queue
            if total_waiting > 0:
                queue_overview = (
                    f"The job scheduling queue currently holds {total_waiting} waiting jobs with varying resource demands and priorities. "
                    f"These jobs have accumulated an average wait time of {avg_wait:.1f} seconds, indicating the current level of system congestion and scheduling pressure. "
                    f"The longest waiting job has been queued for {max_wait:.1f} seconds, which is {max_wait/60:.1f} minutes, while the most recently arrived job has waited only {min_wait:.1f} seconds. "
                    f"Collectively, these pending workloads demand {total_cpu_demand} CPU cores and require {total_power_demand:.1f} watts of power capacity to execute. "
                    f"The scheduler must intelligently balance resource allocation across these competing jobs while considering both performance objectives such as minimizing wait time and bounded slowdown, "
                    f"as well as sustainability goals including maximizing renewable energy utilization and reducing carbon emissions. "
                )
            else:
                queue_overview = (
                    f"The job queue is currently empty with zero pending workloads, indicating that all submitted jobs have been successfully scheduled and allocated to compute resources. "
                    f"This state represents an opportunity for the system to immediately accept and schedule any newly arriving jobs without queueing delays. "
                )
            
            queue_sentences = [queue_overview]
            
            # Mô tả chi tiết từng job trong queue
            for i, job in enumerate(self.job_queue):
                if i >= MAX_QUEUE_SIZE: 
                    break
                
                # Tính toán các metrics cơ bản
                wait_time = self.current_timestamp - job.submit_time
                green_state, green_val = self.cluster.getGreenJobState(job, self.current_timestamp, currentSlot)
                can_alloc = self.cluster.can_allocated(job)
                power_density = job.power / max(1, job.request_number_of_processors)
                
                cpu_demand_ratio = (job.request_number_of_processors / max(1, total_cpus)) * 100
                power_demand_ratio = (job.power / max(1, total_power_capacity)) * 100 if total_power_capacity > 0 else 0
                execution_energy = job.power * job.request_time
                
                # Ngữ cảnh về thời gian chờ và ưu tiên
                if wait_time > 600:  # 10 phút
                    wait_context = (
                        f"Job number {i} in the queue has experienced critical queueing delay of {wait_time:.0f} seconds, equivalent to {wait_time/60:.1f} minutes, "
                        f"which significantly exceeds acceptable service level agreements and indicates severe resource contention or suboptimal scheduling decisions. "
                        f"This extended wait time has accumulated substantial scheduling penalty and priority, making this job a high-priority candidate for immediate allocation to prevent further performance degradation. "
                    )
                elif wait_time > 300:  # 5 phút
                    wait_context = (
                        f"Job number {i} has been waiting in the queue for {wait_time:.0f} seconds, approximately {wait_time/60:.1f} minutes, "
                        f"representing moderate scheduling latency that requires attention to maintain system throughput and user satisfaction. "
                        f"The accumulated wait penalty suggests this job should be prioritized relative to more recently submitted workloads. "
                    )
                elif wait_time > 120:  # 2 phút
                    wait_context = (
                        f"Job number {i} has experienced {wait_time:.0f} seconds of wait time, which is {wait_time/60:.1f} minutes, "
                        f"falling within normal scheduling variance for a busy cluster but still requiring timely allocation to prevent queue buildup. "
                    )
                else:
                    wait_context = (
                        f"Job number {i} recently entered the scheduling queue only {wait_time:.0f} seconds ago, "
                        f"indicating it is a newly submitted workload with minimal accumulated wait penalty and lower scheduling urgency compared to longer-waiting jobs. "
                    )
                
                # Ngữ cảnh về yêu cầu tài nguyên và đặc điểm workload
                resource_context = (
                    f"This job requires {job.request_number_of_processors} CPU cores, representing {cpu_demand_ratio:.1f} percent of the cluster's total {total_cpus} core capacity, "
                    f"and will consume {job.power:.1f} watts of power accounting for {power_demand_ratio:.1f} percent of available power budget during execution. "
                    f"The workload is scheduled to run for {job.request_time:.0f} seconds, which is {job.request_time/60:.1f} minutes or {job.request_time/3600:.2f} hours, "
                    f"and will consume a total of {execution_energy:.1f} watt-seconds of energy throughout its execution lifecycle. "
                )
                
                # Phân loại workload dựa trên power density
                if power_density > 50:
                    workload_context = (
                        f"With a power density of {power_density:.2f} watts per core, this workload exhibits extremely high computational intensity characteristic of GPU-accelerated applications, "
                        f"deep neural network training, computational fluid dynamics simulations, or high-performance numerical computing that stresses both processing units and power infrastructure. "
                        f"Such power-intensive jobs often require specialized cooling, may trigger thermal management protocols, and represent significant opportunities for carbon reduction through intelligent timing with renewable energy availability. "
                    )
                elif power_density > 20:
                    workload_context = (
                        f"The power density of {power_density:.2f} watts per core indicates moderate computational intensity typical of data analytics, scientific computing, parallel processing tasks, "
                        f"or traditional HPC applications that balance CPU utilization with memory and I/O operations. "
                        f"These workloads offer flexibility in scheduling decisions as they neither dominate power budgets nor represent minimal energy consumption. "
                    )
                else:
                    workload_context = (
                        f"At a power density of {power_density:.2f} watts per core, this job demonstrates low energy intensity suggesting it is memory-bound, I/O-intensive, "
                        f"or involves lightweight distributed computing such as data transformation, web services, or embarrassingly parallel tasks with minimal computational requirements per core. "
                        f"These jobs have minimal impact on power budgets and thermal management, making them excellent candidates for backfilling during periods of high system utilization. "
                    )
                
                # Ngữ cảnh về khả năng phân bổ
                if can_alloc:
                    available_cpus = free_cpus
                    cpu_usage_if_alloc = (job.request_number_of_processors / max(1, available_cpus)) * 100 if available_cpus > 0 else 100
                    
                    alloc_context = (
                        f"The cluster currently possesses sufficient free resources to allocate this job immediately without delays, with {available_cpus} CPU cores available "
                        f"from the total capacity of {total_cpus} cores. "
                        f"Executing this job now would consume {cpu_usage_if_alloc:.1f} percent of currently free CPU resources, "
                        f"potentially creating opportunities for smaller jobs to backfill remaining capacity or alternatively constraining resources for future high-priority arrivals. "
                        f"The scheduler must weigh the benefits of immediate allocation, which reduces this job's wait time and improves its individual performance metrics, "
                        f"against the potential opportunity cost of reserving resources for workloads with better renewable energy alignment or higher system-level efficiency. "
                    )
                else:
                    cpu_shortage = max(0, job.request_number_of_processors - free_cpus)
                    
                    blocking_context = (
                        f"This job is currently blocked from execution due to insufficient available resources, specifically requiring {cpu_shortage} additional CPU cores "
                        f"beyond the {free_cpus} cores currently free from the total {total_cpus} core capacity. "
                        f"The scheduler must wait for currently executing jobs to complete and release their allocated resources before this job can proceed. "
                    )
                    
                    # Dự đoán thời gian có thể allocate
                    if self.running_jobs:
                        self.running_jobs.sort(key=lambda rj: rj.scheduled_time + rj.request_time)
                        cumulative_freed = 0
                        estimated_wait = 0
                        for running_job in self.running_jobs:
                            completion_time = running_job.scheduled_time + running_job.request_time - self.current_timestamp
                            cumulative_freed += running_job.request_number_of_processors
                            if cumulative_freed >= cpu_shortage:
                                estimated_wait = completion_time
                                break
                        
                        if estimated_wait > 0:
                            alloc_context = (
                                f"{blocking_context}"
                                f"Based on the current execution schedule of running jobs, sufficient resources are estimated to become available in approximately {estimated_wait:.1f} seconds, "
                                f"which is {estimated_wait/60:.1f} minutes from now, when enough jobs complete to free the required {cpu_shortage} additional cores. "
                                f"This projection helps the scheduler evaluate whether to defer resource-compatible jobs to optimize for renewable energy windows within this timeframe. "
                            )
                        else:
                            alloc_context = blocking_context
                    else:
                        alloc_context = (
                            f"{blocking_context}"
                            f"However, there are currently no running jobs, which indicates a potential system state inconsistency that requires investigation. "
                        )
                
                # Ngữ cảnh về năng lượng xanh và carbon footprint
                renewable_ratio = green_state
                carbon_saving = green_val
                
                if renewable_ratio > 0.8:
                    green_context = (
                        f"The renewable energy assessment for this job shows exceptionally favorable conditions with green energy availability at {renewable_ratio*100:.1f} percent of grid capacity, "
                        f"meaning that if scheduled now, this job would execute primarily on clean energy from solar, wind, and hydroelectric sources. "
                        f"With a carbon saving ratio of {carbon_saving:.4f}, executing this job during the current time window would result in minimal carbon emissions "
                        f"equivalent to approximately {(1-carbon_saving)*100:.1f} percent of peak fossil fuel emissions for equivalent work. "
                        f"This represents an optimal environmental scheduling opportunity where the datacenter can advance both performance and sustainability objectives simultaneously, "
                        f"making immediate allocation highly desirable from a carbon-aware computing perspective. "
                    )
                elif renewable_ratio > 0.5:
                    green_context = (
                        f"Renewable energy penetration currently stands at {renewable_ratio*100:.1f} percent, indicating that over half of the grid power supply derives from clean sources, "
                        f"which creates moderately favorable conditions for carbon-conscious scheduling. "
                        f"The carbon saving ratio of {carbon_saving:.4f} suggests that executing this job now would produce approximately {carbon_saving*100:.1f} percent less carbon emissions "
                        f"compared to execution during peak fossil fuel dependency periods. "
                        f"While not optimal, this window offers reasonable environmental performance, and scheduling decisions should balance the {wait_time:.1f} second accumulated wait time "
                        f"against the potential for marginally better renewable energy conditions in near-term future time windows. "
                    )
                elif renewable_ratio > 0.2:
                    green_context = (
                        f"Current renewable energy availability is limited at only {renewable_ratio*100:.1f} percent of total grid capacity, "
                        f"indicating that the power grid relies predominantly on conventional fossil fuel generation including coal and natural gas at this time. "
                        f"With a carbon saving ratio of {carbon_saving:.4f}, executing this job now would still produce substantial carbon emissions, "
                        f"specifically approximately {(1-carbon_saving)*execution_energy:.1f} watt-seconds of brown energy consumption from the total {execution_energy:.1f} watt-second energy requirement. "
                        f"From an environmental perspective, this represents a suboptimal scheduling window, and carbon-aware policies should seriously consider deferring this job "
                        f"if forecast data indicates significantly higher renewable availability within a timeframe that does not excessively compromise this job's wait time and performance metrics. "
                    )
                else:
                    green_context = (
                        f"The renewable energy situation is critically poor with only {renewable_ratio*100:.1f} percent clean energy in the current grid mix, "
                        f"meaning this job would execute almost entirely on carbon-intensive fossil fuel generation if scheduled now. "
                        f"The extremely low carbon saving ratio of {carbon_saving:.4f} indicates that {(1-carbon_saving)*100:.1f} percent of this job's energy consumption would derive from brown sources, "
                        f"resulting in substantial CO2 emissions estimated at {(1-carbon_saving)*execution_energy:.1f} watt-seconds of fossil fuel energy from the {execution_energy:.1f} watt-second total requirement. "
                        f"This represents an environmentally unfavorable scheduling window where aggressive carbon-aware scheduling policies would strongly recommend deferring execution, "
                        f"potentially achieving up to {(renewable_ratio - 0.8)*100:.1f} percent carbon reduction by waiting for forecast renewable energy peaks, "
                        f"though this must be carefully balanced against the accumulated {wait_time:.1f} second wait time and fairness considerations for this job. "
                    )
                
                # Ngữ cảnh về quyết định scheduling
                priority_score = wait_time * job.power
                efficiency_ratio = job.request_time / max(1, job.power)
                
                if efficiency_ratio > 10:
                    scheduling_context = (
                        f"This job exhibits an efficiency ratio of {efficiency_ratio:.2f} seconds per watt, characterizing it as a long-running batch workload "
                        f"relative to its power consumption, typical of overnight simulations, extended data processing pipelines, or sustained scientific computations. "
                        f"Such jobs offer maximum flexibility for intelligent scheduling because their extended runtime allows the scheduler to time execution strategically within renewable energy windows "
                        f"without significantly impacting job completion time, making them ideal candidates for carbon-aware deferral policies that can reduce emissions by 20-40 percent "
                        f"through optimal temporal placement aligned with solar generation peaks or wind availability patterns. "
                    )
                elif efficiency_ratio > 1:
                    scheduling_context = (
                        f"With an efficiency ratio of {efficiency_ratio:.2f} seconds per watt, this represents a balanced workload where execution duration and energy consumption are proportional, "
                        f"characteristic of standard interactive applications, medium-duration analytical tasks, or typical HPC workloads. "
                        f"These jobs offer moderate scheduling flexibility, allowing for some renewable energy optimization through tactical delays measured in minutes rather than hours, "
                        f"while still maintaining reasonable responsiveness and user satisfaction. "
                    )
                else:
                    scheduling_context = (
                        f"The efficiency ratio of {efficiency_ratio:.2f} seconds per watt reveals this is a power-hungry burst workload that prioritizes speed over energy efficiency, "
                        f"possibly involving intensive parallel processing, GPU acceleration, real-time computing requirements, or short-duration high-performance tasks "
                        f"where users expect immediate turnaround and minimal queueing delay. "
                        f"Such jobs offer limited scheduling flexibility for carbon optimization as deferral directly impacts user-perceived performance, "
                        f"suggesting that resource availability rather than renewable energy timing should dominate allocation decisions unless renewable conditions are exceptionally favorable. "
                    )
                
                summary_context = (
                    f"The accumulated scheduling priority score stands at {priority_score:.2f}, calculated as wait time multiplied by power requirements, "
                    f"which provides a composite metric balancing fairness through wait time consideration with system impact through power consumption weighting. "
                    f"{scheduling_context}"
                )
                
                # Kết hợp tất cả ngữ cảnh
                job_desc = (
                    f"{wait_context}"
                    f"{resource_context}"
                    f"{workload_context}"
                    f"{alloc_context}"
                    f"{green_context}"
                    f"{summary_context}"
                )
                
                queue_sentences.append(job_desc)
            
            queue_text = " ".join(queue_sentences)
            
            # --------------------------------------------------------------------------
            # PART 2: RUNNING JOBS & CLUSTER CONTEXT
            # --------------------------------------------------------------------------
            num_running = len(self.running_jobs)
            node_utilization = ((self.cluster.total_node - self.cluster.free_node) / max(1, self.cluster.total_node)) * 100
            
            if self.running_jobs:
                sorted_by_completion = sorted(self.running_jobs, key=lambda j: j.scheduled_time + j.request_time)
                next_completion_time = sorted_by_completion[0].scheduled_time + sorted_by_completion[0].request_time - self.current_timestamp
                avg_remaining_time = sum((j.scheduled_time + j.request_time - self.current_timestamp) for j in self.running_jobs) / num_running
                total_power_consuming = sum(j.power for j in self.running_jobs)
            else:
                next_completion_time = 0
                avg_remaining_time = 0
                total_power_consuming = 0
            
            # Câu mở đầu cluster context
            if num_running > 0:
                cluster_overview = (
                    f"The datacenter cluster is currently operating at {node_utilization:.1f} percent node utilization with {self.cluster.used_node} compute nodes "
                    f"actively executing workloads from the total {self.cluster.total_node} node capacity, leaving {self.cluster.free_node} nodes in idle state available for new allocations. "
                    f"There are {num_running} jobs running concurrently across the cluster, collectively consuming {used_cpus} CPU cores which represents {cpu_utilization:.1f} percent "
                    f"of the total {total_cpus} core computational capacity, while drawing {total_power_consuming:.1f} watts of instantaneous power. "
                    f"The scheduler currently has {free_cpus} free CPU cores available in the resource pool for immediate allocation to waiting jobs. "
                    f"The next job completion event is projected to occur in {next_completion_time:.1f} seconds, approximately {next_completion_time/60:.1f} minutes from now, "
                    f"which will trigger resource release and create new scheduling opportunities for queued workloads. "
                    f"On average across all running jobs, there are {avg_remaining_time:.1f} seconds, or {avg_remaining_time/60:.1f} minutes, of execution time remaining "
                    f"before jobs complete and free their allocated resources, providing insight into near-term resource availability dynamics. "
                )
            else:
                cluster_overview = (
                    f"The cluster is currently operating in a completely idle state with zero active job executions across all {self.cluster.total_node} available compute nodes. "
                    f"The entire capacity of {total_cpus} CPU cores stands ready for immediate allocation without any resource contention. "
                    f"This idle condition represents an exceptional scheduling opportunity where the system can instantly accept and begin executing any waiting jobs "
                    f"without backfilling considerations or resource conflicts, though it may also indicate potential system underutilization if jobs remain queued, "
                    f"suggesting the scheduler should aggressively allocate pending workloads to maximize cluster efficiency and throughput. "
                )
            
            running_sentences = [cluster_overview]
            
            # Mô tả chi tiết running jobs
            sorted_running = sorted(self.running_jobs, key=lambda j: j.scheduled_time + j.request_time)
            for i, job in enumerate(sorted_running):
                if i >= run_win: 
                    break
                
                remaining_time = (job.scheduled_time + job.request_time) - self.current_timestamp
                elapsed_time = self.current_timestamp - job.scheduled_time
                progress = (elapsed_time / max(1, job.request_time)) * 100
                power_density = job.power / max(1, job.request_number_of_processors)
                cpu_fraction = (job.request_number_of_processors / max(1, total_cpus)) * 100
                
                # Ngữ cảnh completion timeline
                if remaining_time < 60:
                    completion_context = (
                        f"Running job {i} is in its final execution phase with only {remaining_time:.1f} seconds remaining before completion, less than one minute, "
                        f"having already executed for {elapsed_time:.1f} seconds representing {progress:.1f} percent progress through its {job.request_time:.0f} second allocated runtime. "
                        f"This job will imminently release {job.request_number_of_processors} CPU cores back to the free resource pool, "
                        f"creating immediate near-term scheduling opportunities for waiting workloads within the next minute. "
                    )
                elif remaining_time < 300:
                    completion_context = (
                        f"Running job {i} is approaching completion with {remaining_time:.1f} seconds left, approximately {remaining_time/60:.1f} minutes, "
                        f"having already completed {progress:.1f} percent of its {job.request_time:.0f} second workload after running for {elapsed_time:.1f} seconds. "
                        f"Resource release is anticipated in the near future, and the scheduler should proactively consider this in allocation planning for jobs requiring {job.request_number_of_processors} or fewer cores. "
                    )
                elif remaining_time < 600:
                    completion_context = (
                        f"Running job {i} is in mid-execution with {remaining_time:.1f} seconds remaining, about {remaining_time/60:.1f} minutes, "
                        f"currently at {progress:.1f} percent completion after {elapsed_time:.1f} seconds of runtime from its total {job.request_time:.0f} second allocation. "
                        f"This job will continue occupying its {job.request_number_of_processors} cores for a moderate duration before releasing resources. "
                    )
                else:
                    completion_context = (
                        f"Running job {i} is still in early execution stages with {remaining_time:.1f} seconds remaining, which is {remaining_time/60:.1f} minutes or {remaining_time/3600:.2f} hours. "
                        f"At {progress:.1f} percent completion after {elapsed_time:.1f} seconds of runtime from its {job.request_time:.0f} second total allocation, "
                        f"this represents a long-running workload that will hold {job.request_number_of_processors} cores for an extended period, "
                        f"significantly constraining available cluster capacity and potentially extending queue wait times for resource-intensive pending jobs. "
                    )
                
                # Ngữ cảnh resource occupation
                resource_context = (
                    f"This executing job currently occupies {job.request_number_of_processors} CPU cores, accounting for {cpu_fraction:.1f} percent of cluster capacity, "
                    f"while actively consuming {job.power:.1f} watts of power with a density of {power_density:.2f} watts per core. "
                )
                
                # Scheduling impact context
                if i == 0:
                    impact_context = (
                        f"As the next job scheduled to complete, this workload is critically important for scheduling decisions because its imminent completion at {next_completion_time:.1f} seconds "
                        f"will release {job.request_number_of_processors} cores and {job.power:.1f} watts back to the available pool. "
                        f"The scheduler should anticipate this resource release when evaluating allocation feasibility for queued jobs that currently cannot fit, "
                        f"potentially enabling strategic deferral of smaller jobs to align their execution with this predicted resource availability rather than fragmenting current capacity. "
                    )
                elif remaining_time < avg_remaining_time:
                    impact_context = (
                        f"This job will complete sooner than the average running job by approximately {avg_remaining_time - remaining_time:.1f} seconds, "
                        f"contributing to near-term resource availability and helping reduce queue wait times for pending workloads that require its {job.request_number_of_processors} core allocation. "
                    )
                else:
                    impact_context = (
                        f"This job has a longer remaining runtime of {remaining_time - avg_remaining_time:.1f} seconds compared to other running jobs, "
                        f"meaning its occupied {job.request_number_of_processors} cores will remain unavailable for an extended period, "
                        f"potentially prolonging queue delays for resource-constrained workloads and suggesting the scheduler should consider backfilling strategies for smaller jobs. "
                    )
                
                run_desc = (
                    f"{completion_context}"
                    f"{resource_context}"
                    f"{impact_context}"
                )
                
                running_sentences.append(run_desc)
            
            # Trường hợp idle đã được xử lý ở cluster_overview
            running_text = " ".join(running_sentences)
            
            # --------------------------------------------------------------------------
            # PART 3: ENERGY CONTEXT
            # --------------------------------------------------------------------------
            green_slots = self.cluster.greenPower.getGreenPowerSlot(self.current_timestamp)
            
            if green_slots:
                total_forecast_duration = sum(slot['lastTime'] for slot in green_slots[:green_win])
                avg_green_power = sum(slot['power'] for slot in green_slots[:green_win]) / min(len(green_slots), green_win)
                max_green_power = max(slot['power'] for slot in green_slots[:green_win])
                min_green_power = min(slot['power'] for slot in green_slots[:green_win])
                power_variability = max_green_power - min_green_power
            else:
                total_forecast_duration = avg_green_power = max_green_power = min_green_power = power_variability = 0
            
            green_sentences = []
            
            # Energy forecast overview
            if green_slots:
                forecast_overview = (
                    f"The renewable energy forecast provides predictive visibility across the next {total_forecast_duration:.0f} seconds, spanning {total_forecast_duration/3600:.2f} hours "
                    f"through {min(len(green_slots), green_win)} discrete time windows, delivering critical information for carbon-aware scheduling optimization. "
                    f"Average renewable energy availability across this forecast horizon is projected at {avg_green_power:.1f} watts, "
                    f"with significant temporal variability ranging from {min_green_power:.1f} watts during low renewable generation periods "
                    f"to {max_green_power:.1f} watts during peak clean energy production, representing a {power_variability:.1f} watt dynamic range. "
                    f"This variability of {(power_variability/avg_green_power*100):.1f} percent around the mean indicates substantial opportunity for intelligent workload timing, "
                    f"where the scheduler can strategically align energy-intensive computations with renewable energy peaks to minimize datacenter carbon footprint, "
                    f"potentially achieving 20-50 percent emission reductions for flexible batch workloads through optimal temporal placement within forecast windows. "
                )
            else:
                forecast_overview = (
                    f"Renewable energy forecast data is currently unavailable, severely limiting the scheduler's capability to make informed carbon-aware decisions. "
                    f"Without predictive visibility into future grid conditions including solar generation patterns and wind availability, "
                    f"the system must operate reactively based only on instantaneous measurements, missing optimization opportunities that could reduce carbon emissions by 15-35 percent "
                    f"through proactive workload scheduling aligned with anticipated renewable energy peaks in the coming hours. "
                )
            
            green_sentences.append(forecast_overview)
            
            # Detailed time window analysis
            for i, slot in enumerate(green_slots):
                if i >= green_win:
                    break
                
                duration = slot['lastTime']
                total_power = slot['power']
                
                # Tính time offset
                time_offset = sum(green_slots[j]['lastTime'] for j in range(i))
                window_start = time_offset
                window_end = time_offset + duration
                
                # Temporal context
                if i == 0:
                    time_context = (
                        f"Energy forecast window {i} represents the immediate present and near future, beginning now and extending for {duration:.0f} seconds or {duration/60:.1f} minutes. "
                        f"This is the most critical window for scheduling decisions because jobs allocated in the current timestep will execute within this energy profile, "
                        f"making the renewable energy characteristics of this window directly impact the carbon footprint of immediate allocation choices. "
                    )
                elif i < 3:
                    time_context = (
                        f"Energy forecast window {i} covers the time period from {window_start:.0f} seconds, which is {window_start/60:.1f} minutes from now, "
                        f"extending for {duration:.0f} seconds until {window_end:.0f} seconds ahead, representing {window_end/60:.1f} minutes into the future. "
                        f"This near-term window enables tactical scheduling where jobs with moderate execution times of {duration/60:.1f} minutes can be strategically timed "
                        f"to execute entirely within favorable renewable energy conditions if their start can be deferred by {window_start/60:.1f} minutes. "
                    )
                else:
                    time_context = (
                        f"Energy forecast window {i} represents longer-term predictions spanning from {window_start:.0f} seconds to {window_end:.0f} seconds ahead, "
                        f"which is {window_start/3600:.2f} to {window_end/3600:.2f} hours into the future. "
                        f"This extended horizon enables strategic planning for long-running batch workloads and extended simulations that can be scheduled "
                        f"to align their execution windows with predicted renewable energy peaks hours in advance, achieving maximum carbon reduction "
                        f"while maintaining acceptable job completion times. "
                    )
                
                # Renewable energy characterization
                renewable_ratio = min(1.0, total_power / 10000)  # Giả định 10000W là max capacity
                
                if renewable_ratio > 0.8:
                    renewable_context = (
                        f"This forecast window exhibits exceptionally high renewable energy penetration with {total_power:.1f} watts of clean power available, "
                        f"representing approximately {renewable_ratio*100:.1f} percent renewable composition in the grid mix. "
                        f"Solar photovoltaic arrays and wind turbines are projected to dominate power generation during this period, "
                        f"with minimal reliance on fossil fuel backup generation. "
                        f"This constitutes an optimal green scheduling window where the datacenter should aggressively prioritize executing high-power workloads, "
                        f"particularly energy-intensive jobs consuming 1000+ watts that have been queued during less favorable renewable periods, "
                        f"to maximize carbon savings and minimize environmental impact. "
                        f"Scheduling a 5000 watt job during this window versus a low renewable period could reduce carbon emissions by 500-1000 kg CO2 equivalent, "
                        f"demonstrating substantial environmental benefits from intelligent temporal workload placement. "
                    )
                elif renewable_ratio > 0.5:
                    renewable_context = (
                        f"Renewable energy availability in this window stands at {total_power:.1f} watts, representing moderate to good conditions "
                        f"with approximately {renewable_ratio*100:.1f} percent clean energy in the grid composition. "
                        f"While not optimal, this window offers reasonable environmental performance where over half the power derives from renewable sources, "
                        f"making it suitable for scheduling medium-priority workloads that balance performance requirements with sustainability objectives. "
                        f"Jobs executed during this window will produce approximately {(1-renewable_ratio)*100:.1f} percent of the carbon emissions "
                        f"compared to execution during peak fossil fuel dependency, offering meaningful though not maximal carbon reduction. "
                    )
                elif renewable_ratio > 0.2:
                    renewable_context = (
                        f"This forecast window shows limited renewable energy at {total_power:.1f} watts, with only {renewable_ratio*100:.1f} percent clean power in the grid mix, "
                        f"indicating heavy reliance on conventional coal and natural gas generation during this period. "
                        f"Scheduling energy-intensive workloads during this window would result in substantial carbon emissions, "
                        f"with approximately {(1-renewable_ratio)*100:.1f} percent of energy consumption derived from fossil fuels. "
                        f"Carbon-aware scheduling policies should discourage allocation of flexible batch jobs during this period, "
                        f"instead reserving this window for urgent real-time workloads with strict latency requirements while deferring discretionary computing "
                        f"to adjacent windows with better renewable characteristics. "
                    )
                else:
                    renewable_context = (
                        f"This window represents critically poor renewable energy conditions with only {total_power:.1f} watts of clean power available, "
                        f"meaning {renewable_ratio*100:.1f} percent renewable penetration and over {(1-renewable_ratio)*100:.1f} percent fossil fuel dependency. "
                        f"The grid is projected to operate primarily on carbon-intensive coal plants and natural gas turbines during this period, "
                        f"potentially during evening hours after solar generation ceases or during low wind conditions. "
                        f"This constitutes an environmentally unfavorable scheduling window where aggressive carbon-aware policies should strongly avoid allocating any deferrable workloads, "
                        f"as even modest 2000 watt jobs executed during this window could produce 300-500 kg additional CO2 emissions "
                        f"compared to execution during optimal renewable periods forecast in adjacent time windows. "
                    )
                
                # Trend analysis and scheduling implications
                if i < len(green_slots) - 1:
                    next_slot = green_slots[i + 1]
                    next_power = next_slot['power']
                    next_renewable_ratio = min(1.0, next_power / 10000)
                    power_change = next_power - total_power
                    change_percent = (power_change / total_power * 100) if total_power > 0 else 0
                    
                    if abs(change_percent) < 10:
                        trend_context = (
                            f"Looking ahead to the subsequent window, renewable energy availability remains relatively stable at {next_power:.1f} watts, "
                            f"changing by only {abs(change_percent):.1f} percent, suggesting consistent grid conditions across this time horizon. "
                            f"This stability indicates scheduling decisions can focus primarily on resource availability and job priorities "
                            f"rather than attempting to optimize for renewable energy timing between these adjacent windows, "
                            f"as the carbon impact difference would be minimal at under 5 percent emission variance. "
                        )
                    elif power_change > 0:
                        trend_context = (
                            f"Renewable energy shows an improving trend with the next window increasing to {next_power:.1f} watts, "
                            f"representing a {change_percent:.1f} percent increase or {power_change:.1f} watt gain in clean power availability. "
                            f"This positive trajectory suggests that flexible jobs with execution times exceeding {duration:.0f} seconds "
                            f"could benefit from strategic deferral by {window_start:.0f} seconds to capture better renewable conditions in the subsequent window, "
                            f"potentially reducing their carbon footprint by {change_percent:.1f} percent through intelligent temporal placement "
                            f"at the cost of {duration:.0f} seconds additional wait time. "
                            f"The scheduler should evaluate this trade-off for low-priority batch workloads where environmental objectives outweigh immediate execution. "
                        )
                    else:
                        trend_context = (
                            f"Renewable energy exhibits a declining trend with the next window dropping to {next_power:.1f} watts, "
                            f"representing a {abs(change_percent):.1f} percent decrease or {abs(power_change):.1f} watt reduction in clean power. "
                            f"This downward trajectory indicates that the current window offers relatively better renewable conditions compared to near-term future periods, "
                            f"suggesting that jobs which can execute within this {duration:.0f} second window should be allocated now rather than deferred, "
                            f"as waiting would result in {abs(change_percent):.1f} percent higher carbon emissions during the subsequent lower renewable period. "
                            f"This creates urgency for scheduling carbon-intensive workloads that fit within the current window's duration. "
                        )
                else:
                    trend_context = (
                        f"This represents the final window in the current renewable energy forecast horizon at {window_end:.0f} seconds ahead, "
                        f"which is {window_end/3600:.2f} hours into the future, marking the planning limit for proactive carbon-aware scheduling decisions. "
                        f"Jobs with execution times extending beyond this forecast boundary must be evaluated based on historical renewable patterns "
                        f"rather than predictive data, introducing uncertainty in carbon impact assessments for very long-running workloads exceeding {window_end/3600:.1f} hour duration. "
                    )
                
                slot_desc = (
                    f"{time_context}"
                    f"{renewable_context}"
                    f"{trend_context}"
                )
                
                green_sentences.append(slot_desc)
            
            green_text = " ".join(green_sentences)
            
            # --------------------------------------------------------------------------
            # ENCODING với Sentence Transformer
            # --------------------------------------------------------------------------
            embeddings = self.text_encoder.encode(
                [queue_text, running_text, green_text], 
                show_progress_bar=False, 
                batch_size=3,
                convert_to_numpy=True,
                device=self.text_encoder.device
            )
            
            return embeddings.flatten()

        elif repre == "semi-text":
            pass

    # @profile
    def moveforward_for_resources_backfill(self, job):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        for running_job in self.running_jobs:
            free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if free_processors >= job.request_number_of_processors:
                break

        while not self.cluster.can_allocated(job):
            # try to backfill as many jobs as possible
            if self.backfill==1:
                self.job_queue.sort(key=lambda _j: self.backfill_score(_j))
            else:
                self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
            job_queue_iter_copy = list(self.job_queue)

            for _j in job_queue_iter_copy:
                if  _j!=job and (self.current_timestamp + _j.request_time) < earliest_start_time and self.cluster.backfill_check(self.running_jobs, _j, self.current_timestamp, self.backfill):
                    # we should be OK to schedule the job now
                    assert _j.scheduled_time == -1  # this job should never be scheduled before.
                    _j.scheduled_time = self.current_timestamp
                    _j.allocated_machines = self.cluster.allocate(_j.job_id, _j.request_number_of_processors)
                    self.cluster.PowerStruc.update(_j.scheduled_time,
                                                   _j.scheduled_time + _j.run_time,
                                                   _j.power)
                    self.running_jobs.append(_j)
                    score = self.job_score(_j)  # calculated reward
                    self.scheduled_rl[_j.job_id] = score
                    self.job_queue.remove(_j)  # remove the job from job queue

            # move to the next timestamp
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            nextGreenChange = ((self.current_timestamp // 3600) + 1) * 3600
            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= min(next_resource_release_time,nextGreenChange):
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            elif nextGreenChange < next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, nextGreenChange)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job
        self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))

    def skip_for_resources(self, job):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        while not self.cluster.can_allocated(job):
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[
                self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    # @profile
    def moveforward_for_job(self):
        if self.job_queue:
            return True

        # if we need to add job, but can not add any more, return False indicating the job_queue is for sure empty now.
        if self.next_arriving_job_idx >= self.last_job_in_batch:
            assert not self.job_queue
            return False

        # move forward to add jobs into job queue.
        while not self.job_queue:
            if not self.running_jobs:  # there are no running jobs
                next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                next_resource_release_machines = []
            else:
                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
                return True  # job added
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    def job_score(self, job_for_scheduling):

        _tmp = max(1.0, (float(
            job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                         /
                         max(job_for_scheduling.run_time, 10)))
        return _tmp

    def has_only_one_job(self):
        if len(self.job_queue) == 1:
            return True
        else:
            return False

    def schedule(self, job_for_scheduling):
        # make sure we move forward and release needed resources
        if not self.cluster.can_allocated(job_for_scheduling):
            self.skip_for_resources(job_for_scheduling)

        # we should be OK to schedule the job now
        assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
        job_for_scheduling.scheduled_time = self.current_timestamp
        job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                      job_for_scheduling.request_number_of_processors)
        self.cluster.PowerStruc.update(job_for_scheduling.scheduled_time,
                                       job_for_scheduling.scheduled_time + job_for_scheduling.run_time,
                                       job_for_scheduling.power)
        self.running_jobs.append(job_for_scheduling)
        score = self.job_score(job_for_scheduling)  # calculated reward
        self.scheduled_rl[job_for_scheduling.job_id] = score
        self.job_queue.remove(job_for_scheduling)  # remove the job from job queue

        # after scheduling, check if job queue is empty, try to add jobs.
        not_empty = self.moveforward_for_job()

        if not_empty:
            # job_queue is not empty
            return False
        else:
            # job_queue is empty and can not add new jobs as we reach the end of the sequence
            return True

    def valid(self, a):
        action = a[0]
        return self.pairs[action][0]

    def skip1(self,a2):
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        release_index=a2-1
        release_time = (self.running_jobs[release_index].scheduled_time + self.running_jobs[release_index].request_time)
        skipTime = min(release_time,3600+self.current_timestamp)
        next_time_after_skip = skipTime

        next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
        next_resource_release_machines = []
        next_job_sumbitTime = sys.maxsize
        if self.running_jobs:  # there are running jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

        if self.next_arriving_job_idx < self.last_job_in_batch:
            next_job_sumbitTime=self.loads[self.next_arriving_job_idx].submit_time

        while True:
            if next_time_after_skip < min(next_job_sumbitTime,next_resource_release_time):
                self.current_timestamp = max(self.current_timestamp,next_time_after_skip)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                return
            if next_job_sumbitTime <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp,
                                             self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1

                if self.next_arriving_job_idx < self.last_job_in_batch:
                    next_job_sumbitTime = self.loads[self.next_arriving_job_idx].submit_time
                else:
                    next_job_sumbitTime=sys.maxsize
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.
                if len(self.running_jobs)>0:
                    next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                    next_resource_release_machines = self.running_jobs[0].allocated_machines
                else:
                    next_resource_release_time = sys.maxsize

    def skip2(self, skipTime):
        next_time_after_skip = self.current_timestamp + skipTime

        next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
        next_resource_release_machines = []

        next_job_sumbitTime = sys.maxsize
        if self.running_jobs:  # there are running jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

        if self.next_arriving_job_idx < self.last_job_in_batch:
            next_job_sumbitTime=self.loads[self.next_arriving_job_idx].submit_time

        while True:
            if next_time_after_skip < min(next_job_sumbitTime,next_resource_release_time):
                self.current_timestamp = max(self.current_timestamp,next_time_after_skip)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                return
            if next_job_sumbitTime <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp,
                                             self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1

                if self.next_arriving_job_idx < self.last_job_in_batch:
                    next_job_sumbitTime = self.loads[self.next_arriving_job_idx].submit_time
                else:
                    next_job_sumbitTime=sys.maxsize
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.
                if len(self.running_jobs)>0:
                    next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                    next_resource_release_machines = self.running_jobs[0].allocated_machines
                else:
                    next_resource_release_time = sys.maxsize

    def moveforward_green_backfilling_delay_action1(self, job, a):
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        release_index=a-1
        release_time = (self.running_jobs[release_index].scheduled_time + self.running_jobs[release_index].request_time)
        skipTime = min(release_time,3600+self.current_timestamp)

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        if free_processors < job.request_number_of_processors:
            for running_job in self.running_jobs:
                free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
                earliest_start_time = (running_job.scheduled_time + running_job.request_time)
                if free_processors >= job.request_number_of_processors:
                    break
        earliest_start_time = max(earliest_start_time, skipTime)

        while not self.cluster.can_allocated(job) or self.current_timestamp<skipTime:

            self.job_queue.sort(key=lambda _j: self.backfill_score(_j))
            job_queue_iter_copy = list(self.job_queue)

            for _j in job_queue_iter_copy:
                if _j!=job and (self.current_timestamp + _j.request_time) < earliest_start_time and self.cluster.backfill_check(self.running_jobs, _j, self.current_timestamp, self.backfill):
                    # we should be OK to schedule the job now
                    assert _j.scheduled_time == -1  # this job should never be scheduled before.
                    _j.scheduled_time = self.current_timestamp
                    _j.allocated_machines = self.cluster.allocate(_j.job_id, _j.request_number_of_processors)
                    self.cluster.PowerStruc.update(_j.scheduled_time,
                                                   _j.scheduled_time + _j.run_time,
                                                   _j.power)
                    self.running_jobs.append(_j)
                    score = self.job_score(_j)  # calculated reward
                    self.scheduled_rl[_j.job_id] = score
                    self.job_queue.remove(_j)  # remove the job from job queue

            # move to the next timestamp
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            nextGreenChange = ((self.current_timestamp // 3600) + 1) * 3600
            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= min(next_resource_release_time,nextGreenChange):
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            elif nextGreenChange < next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, nextGreenChange)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job
        self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))

    def moveforward_green_backfilling_delay_action2(self, job, ToskipTime):
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        skipTime = ToskipTime+self.current_timestamp

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        if free_processors < job.request_number_of_processors:
            for running_job in self.running_jobs:
                free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
                earliest_start_time = (running_job.scheduled_time + running_job.request_time)
                if free_processors >= job.request_number_of_processors:
                    break
        earliest_start_time = max(earliest_start_time, skipTime)

        while not self.cluster.can_allocated(job) or self.current_timestamp<skipTime:
            self.job_queue.sort(key=lambda _j: self.backfill_score(_j))
            job_queue_iter_copy = list(self.job_queue)

            for _j in job_queue_iter_copy:
                if  _j!=job and (self.current_timestamp + _j.request_time) < earliest_start_time and self.cluster.backfill_check(self.running_jobs, _j, self.current_timestamp, self.backfill):
                    # we should be OK to schedule the job now
                    assert _j.scheduled_time == -1  # this job should never be scheduled before.
                    _j.scheduled_time = self.current_timestamp
                    _j.allocated_machines = self.cluster.allocate(_j.job_id, _j.request_number_of_processors)
                    self.cluster.PowerStruc.update(_j.scheduled_time,
                                                   _j.scheduled_time + _j.run_time,
                                                   _j.power)
                    self.running_jobs.append(_j)
                    score = self.job_score(_j)  # calculated reward
                    self.scheduled_rl[_j.job_id] = score
                    self.job_queue.remove(_j)  # remove the job from job queue

            next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
            next_resource_release_machines = []
            if self.running_jobs:  # there are running jobs
                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machines

            next_job_sumbitTime = sys.maxsize
            if self.next_arriving_job_idx < self.last_job_in_batch:
                next_job_sumbitTime = self.loads[self.next_arriving_job_idx].submit_time

            nextGreenChange = ((self.current_timestamp // 3600) + 1) * 3600
            if skipTime > self.current_timestamp and skipTime < min(next_job_sumbitTime,next_resource_release_time,nextGreenChange):
                self.current_timestamp = max(self.current_timestamp,skipTime)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
            elif next_job_sumbitTime <= min(next_resource_release_time,nextGreenChange):
                self.current_timestamp = max(self.current_timestamp,
                                             self.loads[self.next_arriving_job_idx].submit_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            elif nextGreenChange < next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, nextGreenChange)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.PowerStruc.updateCurrentTime(self.current_timestamp)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.
        self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))


    def schedule_backfill(self, job_for_scheduling,a2):
        if a2==0:
            if not self.cluster.can_allocated(job_for_scheduling):
                self.moveforward_for_resources_backfill(job_for_scheduling)
        else:
            if a2 >0 and a2<=delayMaxJobNum:
                self.moveforward_green_backfilling_delay_action1(job_for_scheduling, a2)
            elif a2 > delayMaxJobNum:
                ToskipTime = delayTimeList[a2 - delayMaxJobNum - 1]
                self.moveforward_green_backfilling_delay_action2(job_for_scheduling, ToskipTime)

        # we should be OK to schedule the job now
        assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
        job_for_scheduling.scheduled_time = self.current_timestamp
        job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                      job_for_scheduling.request_number_of_processors)
        self.cluster.PowerStruc.update(job_for_scheduling.scheduled_time,
                                       job_for_scheduling.scheduled_time + job_for_scheduling.run_time,
                                       job_for_scheduling.power)
        self.running_jobs.append(job_for_scheduling)
        score = self.job_score(job_for_scheduling)  # calculated reward
        self.scheduled_rl[job_for_scheduling.job_id] = score
        self.job_queue.remove(job_for_scheduling)  # remove the job from job queue

        # after scheduling, check if job queue is empty, try to add jobs.
        not_empty = self.moveforward_for_job()

        if not_empty:
            # job_queue is not empty
            return False
        else:
            # job_queue is empty and can not add new jobs as we reach the end of the sequence
            return True

    def schedule_backfill_EASY(self, job_for_scheduling,a2):
        if a2 > 0 and a2 <= delayMaxJobNum:
            self.skip1(a2)
        elif a2 > delayMaxJobNum:
            skipTime = delayTimeList[a2 - delayMaxJobNum - 1]
            self.skip2(skipTime)
        if not self.cluster.can_allocated(job_for_scheduling):
            self.moveforward_for_resources_backfill(job_for_scheduling)

        # we should be OK to schedule the job now
        assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
        job_for_scheduling.scheduled_time = self.current_timestamp
        job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                      job_for_scheduling.request_number_of_processors)
        self.cluster.PowerStruc.update(job_for_scheduling.scheduled_time,
                                       job_for_scheduling.scheduled_time + job_for_scheduling.run_time,
                                       job_for_scheduling.power)
        self.running_jobs.append(job_for_scheduling)
        score = self.job_score(job_for_scheduling)  # calculated reward
        self.scheduled_rl[job_for_scheduling.job_id] = score
        self.job_queue.remove(job_for_scheduling)  # remove the job from job queue

        # after scheduling, check if job queue is empty, try to add jobs.
        not_empty = self.moveforward_for_job()

        if not_empty:
            # job_queue is not empty
            return False
        else:
            # job_queue is empty and can not add new jobs as we reach the end of the sequence
            return True


    def schedule_LPTPN_sequence_reset(self):
        # schedule the sequence of jobs using LPTPN algorithm.
        scheduled_logs = {}
        while True:
            self.job_queue.sort(key=lambda j: self.lptpn_score(j))
            flag=1
            for job_for_scheduling in self.job_queue:
                if self.cluster.can_allocated(job_for_scheduling) and self.cluster.LPTPN_check(self.running_jobs,
                                                                                                      job_for_scheduling,
                                                                                                      self.current_timestamp):
                    assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
                    job_for_scheduling.scheduled_time = self.current_timestamp
                    job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                                  job_for_scheduling.request_number_of_processors)
                    self.cluster.PowerStruc.update(job_for_scheduling.scheduled_time,
                                                   job_for_scheduling.scheduled_time + job_for_scheduling.run_time,
                                                   job_for_scheduling.power)
                    self.running_jobs.append(job_for_scheduling)
                    score = self.job_score(job_for_scheduling)  # calculated reward
                    scheduled_logs[job_for_scheduling.job_id] = score
                    self.job_queue.remove(job_for_scheduling)
                    flag=0
            not_empty = self.moveforward_for_job()
            if not not_empty:
                break

            if flag:
                if len(self.running_jobs)==0:
                    job_for_scheduling=self.job_queue[0]
                    assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
                    job_for_scheduling.scheduled_time = self.current_timestamp
                    job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,
                                                                                  job_for_scheduling.request_number_of_processors)
                    self.cluster.PowerStruc.update(job_for_scheduling.scheduled_time,
                                                   job_for_scheduling.scheduled_time + job_for_scheduling.run_time,
                                                   job_for_scheduling.power)
                    self.running_jobs.append(job_for_scheduling)
                    score = self.job_score(job_for_scheduling)  # calculated reward
                    scheduled_logs[job_for_scheduling.job_id] = score
                    self.job_queue.remove(job_for_scheduling)
                    if not not_empty:
                        break
                else:
                    self.skip_for_resources_LPTPN(job_for_scheduling, scheduled_logs)


        self.post_process_score(scheduled_logs)
        greenRwd = self.cluster.greenPower.getGreenPowerUtilization(self.cluster.PowerStruc.powerSlotLog)

        self.cluster.reset()
        self.loads.reset()
        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        return scheduled_logs,greenRwd

    def step(self, a1,a2,repre="feature"):
        job_for_scheduling = self.pairs[a1][0]
        if self.backfill==1:
            done=self.schedule_backfill(job_for_scheduling,a2)
        if self.backfill==2:
            done=self.schedule_backfill_EASY(job_for_scheduling,a2)
        elif self.backfill==0:
            if a2 >0 and a2<=delayMaxJobNum:
                self.skip1(a2)
            elif a2 > delayMaxJobNum:
                skipTime = delayTimeList[a2 - delayMaxJobNum - 1]
                self.skip2(skipTime)
            done = self.schedule(job_for_scheduling)

        if not done:
            obs = self.build_observation(repre=repre)
            return [obs, 0, False, 0, 0, 0,len(self.running_jobs),0]
        else:
            self.post_process_score(self.scheduled_rl)
            rl_total = sum(self.scheduled_rl.values())

            rwd = -rl_total
            greenRwd = self.cluster.greenPower.getGreenPowerUtilization(self.cluster.PowerStruc.powerSlotLog)

            return [None, rwd, True, 0, 0, 0,len(self.running_jobs),greenRwd]

    def step_for_ga(self, a1,a2):
        job_for_scheduling = self.job_queue[a1]
        if self.backfill==1:
            done=self.schedule_backfill(job_for_scheduling,a2)
        if self.backfill==2:
            done=self.schedule_backfill_EASY(job_for_scheduling,a2)
        elif self.backfill==0:
            if a2 >0 and a2<=delayMaxJobNum:
                self.skip1(a2)
            elif a2 > delayMaxJobNum:
                skipTime = delayTimeList[a2 - delayMaxJobNum - 1]
                self.skip2(skipTime)
            done = self.schedule(job_for_scheduling)

        if not done:
            obs = self.build_observation()
            return [obs, 0, False,len(self.running_jobs),0]
        else:
            self.post_process_score(self.scheduled_rl)
            rl_total = sum(self.scheduled_rl.values())
            rwd = -rl_total
            greenRwd = self.cluster.greenPower.getGreenPowerUtilization(self.cluster.PowerStruc.powerSlotLog)
            return [None, rwd, True,len(self.running_jobs),greenRwd]


    def job_score1(self, job_for_scheduling):
       # 用于计算适应度
       _tmp = max(1.0, (float(
           job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.request_time)
                        /
                        max(job_for_scheduling.request_time, 10)))
       return _tmp

    def skip_for_resources_ga(self, job, power_stru,free_processors,runningJobs,CurrrentTimestamp):

        while job.request_number_of_processors > free_processors:
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            runningJobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
            next_resource_release_time = (runningJobs[0].scheduled_time + runningJobs[0].request_time)

            CurrrentTimestamp=max(CurrrentTimestamp, next_resource_release_time)
            power_stru.updateCurrentTime(CurrrentTimestamp)
            free_processors+= runningJobs[0].request_number_of_processors
            runningJobs.pop(0)  # remove the first running job.
        return free_processors,CurrrentTimestamp

    def moveforward_for_resources_backfill_ga(self, job, power_stru,jobs_list,free_processors,runningJobs,CurrentTimestamp,scheduled_logs):

        earliest_start_time = CurrentTimestamp
        runningJobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
        free_copy=copy.deepcopy(free_processors)

        for running_job in runningJobs:
            free_copy += runningJobs[0].request_number_of_processors
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if free_copy >= job.request_number_of_processors:
                break

        backfillJobList=copy.deepcopy(jobs_list)
        if self.backfill == 1:
            backfillJobList.sort(key=lambda _j: self.backfill_score(_j))
        else:
            backfillJobList.sort(key=lambda _j: self.fcfs_score(_j))

        while job.request_number_of_processors > free_processors:
            job_queue_iter_copy = list(backfillJobList)

            runningJobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
            for _j in job_queue_iter_copy:
                if _j!=job and (CurrentTimestamp + _j.request_time) < earliest_start_time:
                    if _j.request_number_of_processors <= free_processors and \
                            self.cluster.backfill_check_ga(runningJobs, _j, CurrentTimestamp , self.current_timestamp, self.backfill):
                        # we should be OK to schedule the job now
                        assert _j.scheduled_time == -1  # this job should never be scheduled before.
                        _j.scheduled_time = CurrentTimestamp
                        free_processors-= _j.request_number_of_processors
                        power_stru.update(_j.scheduled_time,
                                                       _j.scheduled_time + _j.request_time,
                                                       _j.power)
                        runningJobs.append(_j)
                        score = self.job_score(_j)  # calculated reward
                        scheduled_logs[_j.job_id] = score
                        jobs_list.remove(_j)  # remove the job from job queue
                        backfillJobList.remove(_j)

            # move to the next timestamp
            assert runningJobs
            runningJobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (runningJobs[0].scheduled_time + runningJobs[0].run_time)

            nextGreenChange = ((CurrentTimestamp // 3600) + 1) * 3600
            if nextGreenChange <  next_resource_release_time:
                CurrentTimestamp = max(CurrentTimestamp, next_resource_release_time)
                power_stru.updateCurrentTime(CurrentTimestamp)
            else:
                CurrentTimestamp = max(CurrentTimestamp, next_resource_release_time)
                power_stru.updateCurrentTime(CurrentTimestamp)
                free_processors += runningJobs[0].request_number_of_processors
                runningJobs.pop(0)  # remove the first running job

        return free_processors,CurrentTimestamp

    def getfitness(self,solution,Temp_power):
        free_processors=self.cluster.free_node * self.cluster.num_procs_per_node
        runningJobs=copy.deepcopy(self.running_jobs)
        CurrentTimestamp =copy.deepcopy(self.current_timestamp)

        jobs_list = copy.deepcopy([self.job_queue[solution[i]] for i in range(len(self.job_queue))])
        scheduled_logs={}
        while len(jobs_list)>0:
            job_for_scheduling = jobs_list[0]

            if job_for_scheduling.request_number_of_processors > free_processors:
                if self.backfill:
                    free_processors,CurrentTimestamp=self.moveforward_for_resources_backfill_ga(job_for_scheduling, Temp_power,jobs_list,free_processors,runningJobs,CurrentTimestamp,scheduled_logs)
                else:
                    free_processors,CurrentTimestamp=self.skip_for_resources_ga(job_for_scheduling, Temp_power,free_processors,runningJobs,CurrentTimestamp)

            assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
            job_for_scheduling.scheduled_time = CurrentTimestamp
            free_processors-=job_for_scheduling.request_number_of_processors
            Temp_power.update(job_for_scheduling.scheduled_time,
                                           job_for_scheduling.scheduled_time + job_for_scheduling.request_time,
                                           job_for_scheduling.power)
            runningJobs.append(job_for_scheduling)
            score = self.job_score1(job_for_scheduling)  # calculated reward
            scheduled_logs[job_for_scheduling.job_id] = score
            jobs_list.remove(job_for_scheduling)

        self.post_process_score(scheduled_logs)
        rl_total = sum(scheduled_logs.values())
        rwd1 = -rl_total
        greenRwd=self.cluster.greenPower.getGreenPowerUtilization(Temp_power.powerSlotLog)

        return rwd1,greenRwd

