"""Read-only runtime instrumentation for SC2 action protocol failures."""
import json
from pathlib import Path
import sys

from google.protobuf.json_format import MessageToDict
from s2clientprotocol import error_pb2 as errors
from s2clientprotocol import query_pb2 as query
from s2clientprotocol import sc2api_pb2 as sc

from bch.arena import Arena, _MixedActions
from bch.v10_suite import DEFAULT_PLAN, _smoke_one_scenario


context = {}
original_macros = Arena._macro_actions
original_reset = Arena.reset_round


def reset(self, layout, opponent):
    result = original_reset(self, layout, opponent)
    context.update(scenario=self.map_name, opponent=opponent)
    if not context.get("version_reported"):
        print("ENGINE", self._controller.ping(), flush=True)
        data = self._controller.data()
        print("ABILITY_DATA", json.dumps([MessageToDict(item) for item in data.abilities.values()
                                         if item.ability_id in (1, 16, 23, 3674, 3794)]), flush=True)
        context["version_reported"] = True
    return result


def macros(self, commands):
    context["game_loop"] = int(self._obs.observation.game_loop)
    context["episode_steps"] = self.episode_steps
    context["units"] = [MessageToDict(unit) for unit in self._obs.observation.raw_data.units]
    context["commands"] = [str(command) for command in commands]
    return original_macros(self, commands)


def actions(self, request):
    merged = sc.RequestAction(actions=[*self.red_actions, *request.actions])
    response = self.controller.actions(merged)
    bad = [(index, errors.ActionResult.Name(result))
           for index, result in enumerate(response.result[:len(self.red_actions)]) if result != errors.Success]
    if bad:
        context["request"] = MessageToDict(merged)
        context["response"] = MessageToDict(response)
        tags = [tag for action in self.red_actions for tag in action.action_raw.unit_command.unit_tags]
        available = self.controller.query(query.RequestQuery(abilities=[
            query.RequestQueryAvailableAbilities(unit_tag=tag) for tag in sorted(set(tags))]))
        context["available_abilities"] = MessageToDict(available)
        print("ACTION_FAILURE", json.dumps(context), flush=True)
        raise RuntimeError(f"SC2 rejected red macro commands: {bad}")
    return response


Arena.reset_round = reset
Arena._macro_actions = macros
_MixedActions.actions = actions
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
print(_smoke_one_scenario(str(DEFAULT_PLAN), sys.argv[1], 101, 8, str(output)), flush=True)
