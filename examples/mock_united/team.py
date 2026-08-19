"""Sample United - the RFL reference team.

The engine calls build_team(ctx) once on match day. Return two players and
(optionally) a manager. This sample wires cheap multimodal LLM brains through
the engine's helper factories; your team may instead implement decide()
entirely yourself - the only contract is the observation/reply schema in
docs/RFL_RULES.md.

Your players receive detections in metres from the robot's onboard
perception and reply with a SKILL (go_to_ball / kick_toward / walk_to /
turn_to / hold) plus an optional spoken message to their teammate. Deciding
which skill, aimed where, IS the game - the robot handles seeing and walking.

ctx = {"engine_version": str,
       "team_index": 0 or 1,
       "config": <your team.yaml, parsed>}
"""


def build_team(ctx):
    from gauntlet.football import make_football_agent, make_football_manager
    cfg = ctx["config"]
    base = ctx["team_index"] * 2
    players = [make_football_agent(cfg["player_model"], base + k, seed=base + k,
                                   prompt=cfg.get("prompt", "football_v2"))
               for k in range(2)]
    manager = None
    if cfg.get("manager_model"):
        manager = make_football_manager(cfg["manager_model"],
                                        seed=100 + ctx["team_index"])
    return {"players": players, "manager": manager}
