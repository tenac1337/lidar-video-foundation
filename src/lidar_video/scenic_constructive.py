"""Scenic distribution bridge for a single, correlated constructive placement."""

import json

from scenic.core.distributions import Distribution

from lidar_video.scene_program import compile_scene_program


class ConstructivePlacement(Distribution):
    def __init__(self, graph_json, randomize=True, jitter=0.004):
        super().__init__(valueType=tuple)
        self.program = compile_scene_program(json.loads(graph_json), jitter=jitter)
        self.randomize = randomize

    def sampleGiven(self, values):
        return self.program.positions(randomize=self.randomize)
