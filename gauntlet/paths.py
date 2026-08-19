"""Repository-relative paths. All assets resolve from the repo root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = ROOT / "third_party" / "unitree_rl_gym"

G1_DIR = THIRD_PARTY / "resources" / "robots" / "g1_description"
G1_XML = G1_DIR / "g1_12dof.xml"
MESH_DIR = G1_DIR / "meshes"
POLICY_PT = THIRD_PARTY / "deploy" / "pre_train" / "g1" / "motion.pt"
DEPLOY_CFG = THIRD_PARTY / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"

PROMPTS = ROOT / "prompts"
COURSES = ROOT / "courses"
DOCS = ROOT / "docs"
RUNS = ROOT / "runs"
ENVELOPE_JSON = ROOT / "gauntlet" / "envelope.json"
