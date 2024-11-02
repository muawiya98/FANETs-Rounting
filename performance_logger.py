import os
import pandas as pd
from config import root_path
class PerformanceLogger:
    def __init__(self):
        """
        this class holds every needed info (in time series lists) about the simulation
        dropped packets for all reasons, and arrived packets, 2e2 delay, and energy consumption.
        """
        self.dropped_packets_no_connection_list = []
        self.dropped_packets_buffer_overflow_list = []
        self.dropped_packets_expired_list = []
        self.dropped_packets_distance_list = []
        self.arrived_packets_list = []
        self.e2e_delay_list = []
        self.pdr = []  # packet dropping ratio
        self.energy_consumption = []
        self.packets_logger = {}  # every key is a packet id and has two values... (Dropped,Arrived) --> (False, False)
        self.data_frame = pd.DataFrame()
        self.save_step = 0
        # initially .

        # every round values
        self.num_dropped_packets_no_connection = 0
        self.num_dropped_packets_buffer_overflow = 0
        self.num_dropped_packets_expiration = 0
        self.num_dropped_packets_distance = 0
        self.e2e_delay = 0
        self.num_arrived = 0

    def calculate_measures(self):
        all_packets = self.num_dropped_packets_no_connection + self.num_dropped_packets_buffer_overflow + \
                      self.num_dropped_packets_expiration + self.num_dropped_packets_distance + \
                      self.num_arrived

        all_dropped_packets = all_packets - self.num_arrived

        pdr = (all_dropped_packets / max(1, all_packets)) * 100

        self.dropped_packets_no_connection_list.append(self.num_dropped_packets_no_connection)
        self.dropped_packets_buffer_overflow_list.append(self.num_dropped_packets_buffer_overflow)
        self.dropped_packets_expired_list.append(self.num_dropped_packets_expiration)
        self.dropped_packets_distance_list.append(self.num_dropped_packets_distance)
        self.arrived_packets_list.append(self.num_arrived)
        self.e2e_delay_list.append(self.e2e_delay)
        self.pdr.append(pdr)
        self.save_step+=1
        new_row = pd.DataFrame([[self.num_dropped_packets_no_connection, self.num_dropped_packets_buffer_overflow, self.num_dropped_packets_expiration, self.num_dropped_packets_distance, self.num_arrived, self.e2e_delay, pdr]], columns=['num_dropped_packets_no_connection', 'num_dropped_packets_buffer_overflow', 'num_dropped_packets_expiration', 'num_dropped_packets_distance', 'num_arrived', 'e2e_delay', 'pdr'])
        self.data_frame = self.data_frame.append(new_row, ignore_index=True) # type: ignore
        if self.save_step%500==0:
            my_path = os.path.join(root_path, "RL_Results")
            os.makedirs(my_path, exist_ok=True)
            self.data_frame.to_csv(os.path.join(my_path, f'measures.csv'), index=False)

        self.num_dropped_packets_no_connection = 0
        self.num_dropped_packets_buffer_overflow = 0
        self.num_dropped_packets_expiration = 0
        self.num_dropped_packets_distance = 0
        self.e2e_delay = 0
        self.num_arrived = 0
