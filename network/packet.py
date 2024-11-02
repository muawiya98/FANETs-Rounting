class Packet:
    def __init__(
        self,
        source,
        packet_counter,
        destination,
        length,
        header_length,
        generation_moment,
        expiration_date,
        id_=None,
        path=None
    ):
        self.length = length
        self.header_length = header_length
        self.generation_moment = generation_moment
        self.source = source
        self.destination = destination
        self.expiration_date = expiration_date
        if id_ is not None:
            self.id = id_
            self.path = path
        else:
            self.id = self.generate_id(packet_counter)
            self.path = []

    def is_valid(self, time):
        return self.source != self.destination and time < self.expiration_date

    def generate_id(self, packet_counter):
        """
        packet id is a combination between the node id and the packet number in this node.
        """
        return str(self.source) + "_" + str(packet_counter)
