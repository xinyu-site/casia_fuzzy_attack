from amb.envs.base_logger import BaseLogger


class VoltageLogger(BaseLogger):

    def get_task_name(self):
        return self.env_args["scenario"]