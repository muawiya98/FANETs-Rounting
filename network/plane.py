class Plane:
    def __init__(self, **kwargs):
        """
        kwargs : dictionary
        ex: {"name": "Pico-phase", "transmission_range": [10, 300], "mobility": [33.4, 66.6]}
        """
        self.name = kwargs["name"]
        self.transmission_range = kwargs["transmission_range"]
        self.mobility = kwargs["mobility"]

    def __eq__(self, other) -> bool:
        assert isinstance(other, Plane), "'other' not instance from PLane"
        return id(self) == id(other)

    def __hash__(self) -> int:
        return id(self)
