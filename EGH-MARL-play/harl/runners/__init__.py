"""Runner registry."""
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.runners.off_policy_ha_runner import OffPolicyHARunner
from harl.runners.off_policy_ma_runner import OffPolicyMARunner
from harl.runners.on_policy_ma_eval_runner import OnPolicyMAEvalRunner
from harl.runners.off_policy_eghn_runner import OffPolicyEghnRunner
from harl.runners.off_policy_eghn_runner import OffPolicyEghnRunner
from harl.runners.on_policy_dataaug_runner import OnPolicyDARunner
from harl.runners.on_policy_hepn_runner import OnPolicyHepnRunner
from harl.runners.off_policy_hmf_runner import OffPolicyHMFRunner
from harl.runners.on_policy_ma_eval_attack_runner import OnPolicyMAAttackRunner
from harl.runners.on_policy_ma_eval_trainattack_runner import OnPolicyMATrainAttackRunner

RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,
    "hatrpo": OnPolicyHARunner,
    "haa2c": OnPolicyHARunner,
    "haddpg": OffPolicyHARunner,
    "hatd3": OffPolicyHARunner,
    "hasac": OffPolicyHARunner,
    "had3qn": OffPolicyHARunner,
    "maddpg": OffPolicyMARunner,
    "matd3": OffPolicyMARunner,
    "mappo": OnPolicyMARunner,
    "eghn_mappo": OnPolicyMARunner,
    "eghnv2_mappo": OnPolicyMARunner,
    "eghn_actor_mappo": OnPolicyMARunner,
    "eghn_critic_mappo": OnPolicyMARunner,
    "eghn_critic_happo": OnPolicyHARunner,
    "eghn_maddpg": OffPolicyEghnRunner,
    "egnn_mappo": OnPolicyMARunner,
    "egnnv2_mappo": OnPolicyMARunner,
    "egnn_mix_mappo": OnPolicyMARunner,
    "egnnv3_mappo": OnPolicyMARunner,
    "egnn_actor_mappo": OnPolicyMARunner,
    "egnn_critic_mappo": OnPolicyMARunner,
    "egnn_critic_happo": OnPolicyHARunner,
    "egnn_maddpg": OffPolicyEghnRunner,
    "hepn_mappo": OnPolicyHepnRunner,
    "gat_mappo": OnPolicyMARunner,
    "gcn_mappo": OnPolicyMARunner,
    "graphsage_mappo": OnPolicyMARunner,
    "hie_mappo": OnPolicyMARunner,
    "hie_critic_mappo": OnPolicyMARunner,
    "hama_mappo": OnPolicyMARunner,
    "hmf_mappo": OnPolicyMARunner,
    "mappo_data_aug": OnPolicyDARunner,
    "hmf": OffPolicyHMFRunner,
}

EVAL_RUNNER_REGISTRY = {
    "egnnv2_mappo": OnPolicyMAEvalRunner,
    "egnn_mappo": OnPolicyMAEvalRunner,
    "egnn_mix_mappo": OnPolicyMAEvalRunner,
}

ATTACK_RUNNER_REGISTRY = {
    "egnnv2_mappo": OnPolicyMAAttackRunner,
    "egnn_mappo": OnPolicyMAAttackRunner,
    "egnn_mix_mappo": OnPolicyMAAttackRunner,
}

TRAINATTACK_RUNNER_REGISTRY = {
    "egnnv2_mappo": OnPolicyMATrainAttackRunner,
    "egnn_mappo": OnPolicyMATrainAttackRunner,
    "egnn_mix_mappo": OnPolicyMATrainAttackRunner,
}
