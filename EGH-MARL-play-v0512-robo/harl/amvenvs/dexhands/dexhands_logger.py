from amb.envs.base_logger import BaseLogger


class DexHandsLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["task"]
