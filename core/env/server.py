
class Edge_Server(object):
    def __init__(self, es_id, capacity, proc_power, occupied):
        assert capacity > 0
        assert proc_power > 0
        
        self.es_id = es_id                  # edge server id
        self.capacity = capacity            # capacity limit for config functions
        self.proc_power = proc_power        # processing power

        self.deployed = []                  # deployed functions
        self.occupied = occupied            # occupied storage
