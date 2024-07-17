from __future__ import annotations

from enum import Enum
from typing import Set, Tuple
from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.state_machine.state_type import VocabularyStateType, PositionalStateType, VALID_STATE_TYPES
from nuplan_extent.planning.training.modeling.models.utils import dequantize_state_token
import numpy as np
import itertools


class StateMachine:
    def __init__(self, max_vocalbulary_size: int):
        self.vocalbulary_state = VocabularyStateType.KILL_TOKEN
        self.positional_state = PositionalStateType.KILL_STATE
        self.frame_index = 0
        self.prev_tracked_object = [[], [], [], []]
        self.current_tracked_object = [[], [], [], []]
        self.current_newborn_object = [[], [], [], []]
        self.killed_object = []
        self.prev_killed_object = []
        self.prev_object_id = None
        self.valid_vocalbulary = np.zeros(max_vocalbulary_size)

    def set_cond_idx_state(self, tokens):
        for token in tokens:
            self.update_state(token)

    def get_avaliable_id_tokens(self):
        self.valid_vocalbulary[:] = 0
        # sort from biggest to smallest, if current index is in biggest (i.e
        # Cyclist), then only allow cyclist tokens
        current_tracked_object_flatten = list(
            itertools.chain.from_iterable(
                self.current_tracked_object))
        prev_tracked_object_flatten = list(
            itertools.chain.from_iterable(
                self.prev_tracked_object))
        sorted(prev_tracked_object_flatten)
        for object_token in prev_tracked_object_flatten:
            if (object_token not in current_tracked_object_flatten) and (
                    object_token not in self.killed_object):
                self.valid_vocalbulary[object_token] = 1
        self.valid_vocalbulary[0] = 1
        return self.valid_vocalbulary

    def get_avaliable_x_tokens(self):
        self.valid_vocalbulary[:] = 0
        self.valid_vocalbulary[VocabularyStateType.X.start:
                               VocabularyStateType.X.end + 1] = 1
        return self.valid_vocalbulary

    def get_avaliable_y_tokens(self):
        self.valid_vocalbulary[:] = 0
        self.valid_vocalbulary[VocabularyStateType.Y.start:
                               VocabularyStateType.Y.end + 1] = 1
        return self.valid_vocalbulary

    def get_avaliable_heading_tokens(self):
        self.valid_vocalbulary[:] = 0
        self.valid_vocalbulary[VocabularyStateType.HEADING.start:
                               VocabularyStateType.HEADING.end + 1] = 1
        return self.valid_vocalbulary

    def get_avaliable_width_tokens(self):
        self.valid_vocalbulary[:] = 0
        self.valid_vocalbulary[VocabularyStateType.WIDTH.start:VocabularyStateType.WIDTH.end+1] = 1
        return self.valid_vocalbulary

    def get_avaliable_length_tokens(self):
        self.valid_vocalbulary[:] = 0
        self.valid_vocalbulary[VocabularyStateType.LENGTH.start:VocabularyStateType.LENGTH.end+1] = 1
        return self.valid_vocalbulary
        
    def update_state(self, token: int):
        for state in VocabularyStateType:
            if token in state:
                self.vocalbulary_state = state
                break

        # update positional state
        if self.vocalbulary_state in VocabularyStateType.BOS_TOKEN:
            self.positional_state = PositionalStateType.KILL_STATE
            # new frame, reset prev_tracked_object and current_tracked_object
            self.prev_tracked_object = self.current_tracked_object
            self.current_tracked_object = [[], [], [], []]
            self.current_newborn_object = [[], [], [], []]
            self.killed_object = []
        elif self.vocalbulary_state in VocabularyStateType.NEWBORN_TOKEN:
            self.positional_state = PositionalStateType.NEWBORN_STATE
        elif self.vocalbulary_state in VocabularyStateType.KILL_TOKEN:
            self.positional_state = PositionalStateType.NORMAL_STATE
            self.prev_killed_object = []
            self.prev_object_id = None
        else:
            pass

        if self.vocalbulary_state in [VocabularyStateType.EGO, VocabularyStateType.VEHICLE,
                                      VocabularyStateType.PEDESTRIAN, VocabularyStateType.BICYCLE]:
            if self.positional_state in PositionalStateType.NORMAL_STATE:
                self.current_tracked_object[self.vocalbulary_state.index].append(
                    token)
                if self.vocalbulary_state not in VocabularyStateType.EGO:
                    self.prev_object_id = token
            elif self.positional_state in PositionalStateType.NEWBORN_STATE:
                self.current_tracked_object[self.vocalbulary_state.index].append(
                    token)
                self.current_newborn_object[self.vocalbulary_state.index].append(
                    token)
            elif self.positional_state == PositionalStateType.KILL_STATE:
                self.killed_object.append(token)
                if token in self.prev_killed_object:
                    self.prev_killed_object.pop(
                        self.prev_killed_object.index(token))
            else:
                raise ValueError(
                    f"Invalid positional state: {self.positional_state}")

        # maintain killed object
        if token in VocabularyStateType.LOCATION_TOKEN:
            x_token, y_token, h_token, _, _ = dequantize_state_token(token)
            if self.positional_state in PositionalStateType.NORMAL_STATE:
                if x_token == VocabularyStateType.X.start or x_token == VocabularyStateType.X.end or y_token == VocabularyStateType.Y.start or y_token == VocabularyStateType.Y.end:
                    if (self.prev_object_id is not None) and (
                            self.prev_object_id not in self.prev_killed_object):
                        self.prev_killed_object.append(self.prev_object_id)

    def possible_next_token(self):

        self.valid_vocalbulary[:] = 0
        location_state = False
        # print(self.vocalbulary_state)
        if self.vocalbulary_state in [VocabularyStateType.EGO, VocabularyStateType.VEHICLE,
                                      VocabularyStateType.PEDESTRIAN, VocabularyStateType.BICYCLE]:
            if self.positional_state in [
                    PositionalStateType.NORMAL_STATE, PositionalStateType.NEWBORN_STATE]:
                self.valid_vocalbulary[VocabularyStateType.X.start:
                                       VocabularyStateType.HEADING.end + 1] = 1
                location_state = True
            elif self.positional_state == PositionalStateType.KILL_STATE:
                # if in killed state and no object killed, then only allow kill
                # token
                max_killed_object = max(
                    self.killed_object) if len(
                    self.killed_object) > 0 else 0
                prev_tracked_object_flatten = list(
                    itertools.chain.from_iterable(
                        self.prev_tracked_object))
                sorted(prev_tracked_object_flatten)
                if len(self.prev_killed_object) == 0:
                    self.valid_vocalbulary[VocabularyStateType.KILL_TOKEN.start:
                                           VocabularyStateType.KILL_TOKEN.end + 1] = 1
                for object_token in prev_tracked_object_flatten:
                    if object_token in self.prev_killed_object:
                        self.valid_vocalbulary[object_token] = 1
                        break
            else:
                raise ValueError(
                    f"Invalid positional state: {self.positional_state}")
        elif self.vocalbulary_state in VocabularyStateType.LOCATION_TOKEN:
            # if the next is class index, then only monotonic increase is
            # allowed
            if self.positional_state in PositionalStateType.NORMAL_STATE:
                # sort from biggest to smallest, if current index is in biggest
                # (i.e Cyclist), then only allow cyclist tokens
                current_tracked_object_flatten = list(
                    itertools.chain.from_iterable(
                        self.current_tracked_object))
                prev_tracked_object_flatten = list(
                    itertools.chain.from_iterable(
                        self.prev_tracked_object))
                sorted(prev_tracked_object_flatten)
                for object_token in prev_tracked_object_flatten:
                    if (object_token not in current_tracked_object_flatten) and (
                            object_token not in self.killed_object):
                        self.valid_vocalbulary[object_token] = 1
                        break
                if self.valid_vocalbulary.sum() == 0:  # end trakced object, then only allow new born token
                    self.valid_vocalbulary[VocabularyStateType.NEWBORN_TOKEN.start:
                                           VocabularyStateType.NEWBORN_TOKEN.end + 1] = 1
            elif self.positional_state in PositionalStateType.NEWBORN_STATE:
                # sort from biggest to smallest, if current index is in biggest
                # (i.e Cyclist), then only allow cyclist tokens
                for index, current_newborn_object_typelist in enumerate(
                        self.current_newborn_object[::-1]):
                    type_index = 3 - index
                    # if we have new born type, then only born token larger
                    # than that type is allowed
                    if len(current_newborn_object_typelist) > 0:
                        if (VocabularyStateType.index_to_state(
                                type_index).end + 1) > (max(current_newborn_object_typelist) + 1):
                            self.valid_vocalbulary[max(
                                current_newborn_object_typelist) + 1] = 1
                        break  # so, break is here
                    else:  # if not, then newborn token is only larger than one than current tacked object
                        if len(self.current_tracked_object[type_index]) > 0:
                            max_tracked_object = max(
                                self.current_tracked_object[type_index])
                            if max_tracked_object < VocabularyStateType.index_to_state(
                                    type_index).end:
                                self.valid_vocalbulary[max_tracked_object + 1] = 1
                        else:  # if no current tracked object, then newborn token can generate from zero
                            self.valid_vocalbulary[VocabularyStateType.index_to_state(
                                type_index).start] = 1
                # end newborn any time
                self.valid_vocalbulary[VocabularyStateType.BOS_TOKEN.start:
                                       VocabularyStateType.BOS_TOKEN.end + 1] = 1
            else:
                raise ValueError(f"Heading state cannot be in kill state")
        elif self.vocalbulary_state in VocabularyStateType.KILL_TOKEN:
            # ego should always exist
            self.valid_vocalbulary[VocabularyStateType.EGO.start:
                                   VocabularyStateType.EGO.end + 1] = 1
        elif self.vocalbulary_state in VocabularyStateType.BOS_TOKEN:
            prev_tracked_object_flatten = list(
                itertools.chain.from_iterable(
                    self.prev_tracked_object))
            sorted(prev_tracked_object_flatten)
            # only allow kill token if no object killed
            if len(self.prev_killed_object) == 0:
                self.valid_vocalbulary[VocabularyStateType.KILL_TOKEN.start:
                                       VocabularyStateType.KILL_TOKEN.end + 1] = 1
            # from small idx to large idx, if the object is killed, then only
            # allow that object, or if the object is not must be killed, then
            # only allow object larger than that
            for object_token in prev_tracked_object_flatten:
                if object_token in self.prev_killed_object:
                    self.valid_vocalbulary[object_token] = 1
                    break
                # self.valid_vocalbulary[object_token] = 1
            # ego always alive
            self.valid_vocalbulary[VocabularyStateType.EGO.start] = 0
        elif self.vocalbulary_state == VocabularyStateType.NEWBORN_TOKEN:
            # begin newborn
            for index, typelist in enumerate(self.current_tracked_object):
                max_type_object_token = max(
                    typelist) + 1 if len(typelist) > 0 else VocabularyStateType.index_to_state(index).start
                if max_type_object_token <= VocabularyStateType.index_to_state(
                        index).end:
                    self.valid_vocalbulary[max_type_object_token] = 1
            # end newborn
            self.valid_vocalbulary[VocabularyStateType.BOS_TOKEN.start:
                                   VocabularyStateType.BOS_TOKEN.end + 1] = 1
        elif self.vocalbulary_state == VocabularyStateType.PAD_TOKEN:
            self.valid_vocalbulary[VocabularyStateType.PAD_TOKEN.start:
                                   VocabularyStateType.PAD_TOKEN.end + 1] = 1
        else:
            raise ValueError(
                f"Invalid vocalbulary state: {self.vocalbulary_state}")
        return self.valid_vocalbulary

    def _validate_state(self):
        if (self.state_type, self.state) not in VALID_STATE_TYPES:
            raise ValueError(
                f"Invalid state type: {self.state_type}, {self.state}")

    def _validate_transitions(self, next_state: Tuple(
            VocabularyStateType, PositionalStateType)):
        if next_state not in self._valid_transitions[(
                self.state_type, self.state)]:
            raise ValueError(
                f"Invalid transition: {self.state_type}, {self.state} -> {next_state}")

    @property
    def _valid_transitions(self):
        return {
            (VocabularyStateType.EGO, PositionalStateType.NORMAL_STATE): [(VocabularyStateType.X, PositionalStateType.NORMAL_STATE)],
            (VocabularyStateType.VEHICLE, PositionalStateType.NORMAL_STATE): [(VocabularyStateType.X, PositionalStateType.NORMAL_STATE)],
            (VocabularyStateType.PEDESTRIAN, PositionalStateType.NORMAL_STATE): [(VocabularyStateType.X, PositionalStateType.NORMAL_STATE)],
            (VocabularyStateType.BICYCLE, PositionalStateType.NORMAL_STATE): [(VocabularyStateType.X, PositionalStateType.NORMAL_STATE)],
            (VocabularyStateType.X, PositionalStateType.NORMAL_STATE): [(VocabularyStateType.Y, PositionalStateType.NORMAL_STATE)],
            (VocabularyStateType.Y, PositionalStateType.NORMAL_STATE): [(VocabularyStateType.HEADING, PositionalStateType.NORMAL_STATE)],
            (VocabularyStateType.HEADING, PositionalStateType.NORMAL_STATE): [
                (VocabularyStateType.EGO, PositionalStateType.NORMAL_STATE),
                (VocabularyStateType.VEHICLE, PositionalStateType.NORMAL_STATE),
                (VocabularyStateType.PEDESTRIAN, PositionalStateType.NORMAL_STATE),
                (VocabularyStateType.BICYCLE, PositionalStateType.NORMAL_STATE),
                (VocabularyStateType.NEWBORN_TOKEN,
                 PositionalStateType.NEWBORN_STATE),
            ],

            (VocabularyStateType.VEHICLE, PositionalStateType.NEWBORN_STATE): [(VocabularyStateType.X, PositionalStateType.NEWBORN_STATE)],
            (VocabularyStateType.PEDESTRIAN, PositionalStateType.NEWBORN_STATE): [(VocabularyStateType.X, PositionalStateType.NEWBORN_STATE)],
            (VocabularyStateType.BICYCLE, PositionalStateType.NEWBORN_STATE): [(VocabularyStateType.X, PositionalStateType.NEWBORN_STATE)],
            (VocabularyStateType.X, PositionalStateType.NEWBORN_STATE): [(VocabularyStateType.Y, PositionalStateType.NEWBORN_STATE)],
            (VocabularyStateType.Y, PositionalStateType.NEWBORN_STATE): [(VocabularyStateType.HEADING, PositionalStateType.NEWBORN_STATE)],
            (VocabularyStateType.HEADING, PositionalStateType.NEWBORN_STATE): [
                (VocabularyStateType.VEHICLE, PositionalStateType.NEWBORN_STATE),
                (VocabularyStateType.PEDESTRIAN, PositionalStateType.NEWBORN_STATE),
                (VocabularyStateType.BICYCLE, PositionalStateType.NEWBORN_STATE),
                (VocabularyStateType.BOS_TOKEN, PositionalStateType.NORMAL_STATE),
            ],

            (VocabularyStateType.EGO, PositionalStateType.KILL_STATE): [
                (VocabularyStateType.VEHICLE, PositionalStateType.KILL_STATE),
                (VocabularyStateType.PEDESTRIAN, PositionalStateType.KILL_STATE),
                (VocabularyStateType.BICYCLE, PositionalStateType.KILL_STATE),
                (VocabularyStateType.KILL_TOKEN, PositionalStateType.KILL_STATE),
            ],
            (VocabularyStateType.VEHICLE, PositionalStateType.KILL_STATE): [
                (VocabularyStateType.VEHICLE, PositionalStateType.KILL_STATE),
                (VocabularyStateType.PEDESTRIAN, PositionalStateType.KILL_STATE),
                (VocabularyStateType.BICYCLE, PositionalStateType.KILL_STATE),
                (VocabularyStateType.KILL_TOKEN, PositionalStateType.KILL_STATE),
            ],
            (VocabularyStateType.PEDESTRIAN, PositionalStateType.KILL_STATE): [
                (VocabularyStateType.PEDESTRIAN, PositionalStateType.KILL_STATE),
                (VocabularyStateType.BICYCLE, PositionalStateType.KILL_STATE),
                (VocabularyStateType.KILL_TOKEN, PositionalStateType.KILL_STATE),
            ],
            (VocabularyStateType.BICYCLE, PositionalStateType.KILL_STATE): [
                (VocabularyStateType.BICYCLE, PositionalStateType.KILL_STATE),
                (VocabularyStateType.KILL_TOKEN, PositionalStateType.KILL_STATE),
            ],
            (VocabularyStateType.KILL_TOKEN, PositionalStateType.KILL_STATE): [
                (VocabularyStateType.EGO, PositionalStateType.NORMAL_STATE),
                (VocabularyStateType.VEHICLE, PositionalStateType.NORMAL_STATE),
                (VocabularyStateType.PEDESTRIAN, PositionalStateType.NORMAL_STATE),
                (VocabularyStateType.BICYCLE, PositionalStateType.NORMAL_STATE),
                (VocabularyStateType.NEWBORN_TOKEN,
                 PositionalStateType.NEWBORN_STATE),
            ],
            (VocabularyStateType.NEWBORN_TOKEN, PositionalStateType.NEWBORN_STATE): [
                (VocabularyStateType.VEHICLE, PositionalStateType.NORMAL_STATE),
                (VocabularyStateType.PEDESTRIAN, PositionalStateType.NORMAL_STATE),
                (VocabularyStateType.BICYCLE, PositionalStateType.NORMAL_STATE),
                (VocabularyStateType.BOS_TOKEN, PositionalStateType.NORMAL_STATE),
            ],
            (VocabularyStateType.PAD_TOKEN, PositionalStateType.NORMAL_STATE): [(VocabularyStateType.PAD_TOKEN, PositionalStateType.NORMAL_STATE)],
            (VocabularyStateType.BOS_TOKEN, PositionalStateType.NORMAL_STATE): [
                (VocabularyStateType.EGO, PositionalStateType.KILL_STATE),
                (VocabularyStateType.VEHICLE, PositionalStateType.KILL_STATE),
                (VocabularyStateType.PEDESTRIAN, PositionalStateType.KILL_STATE),
                (VocabularyStateType.BICYCLE, PositionalStateType.KILL_STATE),
                (VocabularyStateType.KILL_TOKEN, PositionalStateType.KILL_STATE),
            ],
        }
