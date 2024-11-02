from abc import ABC, abstractmethod
from collections import deque
import datetime


class MessageType(ABC):

    def __init__(self):
        self.content = {}
        self.source = -1
        self.destination = -1

    @abstractmethod
    def send_message(self, *args):
        pass


"""
All the classes below inherit form the MessageType class, and they all construct a message based on their 
type and add it to the event handler. 
"""


class HelloMessage(MessageType):

    def send_message(self, m_type, source, destination, position, speed_ray, role, plane, time, event_handler):

        self.source = source
        self.destination = destination
        self.content = {'position': position, 'speed': speed_ray, 'role': role, 'plane': plane,
                        'timestamp': time}

        # args[6] += [args[0], self.source, self.destination, self.content]
        event_handler.event_listener.append([m_type, self.source, self.destination, self.content])


class UtilityMessage(MessageType):

    def send_message(self, *args):
        """
        :param args: type, source, destination, utility, time, eventHandler object.
        """
        self.source = args[1]
        self.destination = args[2]
        self.content = {'utility': args[3], 'timestamp': args[4]}
        # args[5] += [args[0], self.source, self.destination, self.content]
        args[5].event_listener.append([args[0], self.source, self.destination, self.content])


class ClusterHeadDeclaration(MessageType):

    def send_message(self, *args):
        """
        :param args:
        list of arguments: type, source, destination, utility, time, eventHandler object.
        """
        self.source = args[1]
        self.destination = args[2]
        self.content = {'utility': args[3], 'timestamp': args[4]}
        # args[5] += [args[0], self.source, self.destination, self.content]
        args[5].event_listener.append([args[0], self.source, self.destination, self.content])


class JoinClusterRequest(MessageType):

    def send_message(self, *args):
        """
        list of arguments: type, source, destination, time, eventHandler object.
        """
        self.source = args[1]
        self.destination = args[2]
        self.content = {'timestamp': args[3], 'forwarding_node': args[4]}
        # args[4] += [args[0], self.source, self.destination, self.content]
        args[5].event_listener.append([args[0], self.source, self.destination, self.content])


class JoinClusterResponse(MessageType):

    def send_message(self, *args):
        """
        list of arguments: type, source, destination, time, eventHandler object.
        """
        self.source = args[1]
        self.destination = args[2]
        self.content = {'timestamp': args[3]}
        # args[4] += [args[0], self.source, self.destination, self.content]
        args[4].event_listener.append([args[0], self.source, self.destination, self.content])


class PacketMessage(MessageType):

    def send_message(self, m_type, source, destination, packet_id, path, org_destination, size, moment_of_generation,
                     expiration_date, time_stamp, event_handler):

        self.source = source
        self.destination = destination
        self.content = {'size': size,
                        'path': path,
                        'id': packet_id,
                        'moment_of_generation': moment_of_generation,
                        'timestamp': time_stamp,
                        'expiration_date': expiration_date,
                        'original_destination': org_destination,
                        }
        # args[4] += [args[0], self.source, self.destination, self.content]
        event_handler.event_listener.append([m_type, self.source, self.destination, self.content])


class AffirmationMessage(MessageType):

    def send_message(self, m_type, source, destination, packet_id, path,
                     expiration_date, size, time_stamp, event_handler):

        self.source = source
        self.destination = destination
        self.content = {'path': path,
                        'id': packet_id,
                        'timestamp': time_stamp,
                        'expiration_date': expiration_date,
                        'size': size,
                        }
        event_handler.event_listener.append([m_type, self.source, self.destination, self.content])


class Message:

    def __init__(self):

        self.hello = HelloMessage()
        self.utility = UtilityMessage()
        self.ch = ClusterHeadDeclaration()
        self.join = JoinClusterRequest()
        self.response = JoinClusterResponse()
        self.packet = PacketMessage()
        self.affirmation = AffirmationMessage()

    def __call__(self, *args):

        if args[0] == 'hello':
            self.hello.send_message(*args)
        elif args[0] == 'utility':
            self.utility.send_message(*args)
        elif args[0] == 'ch':
            self.ch.send_message(*args)
        elif args[0] == 'join_request':
            self.join.send_message(*args)
        elif args[0] == 'join_response':
            self.response.send_message(*args)
        elif args[0] == 'packet':
            self.packet.send_message(*args)
        elif args[0] == 'affirmation':
            self.affirmation.send_message(*args)
        else:
            print('name message is not define')
