from harl.common.base_logger import BaseLogger


class MuJoCo3dLogger(BaseLogger):
    def get_task_name(self):
        return f"{self.env_args['scenario']}"
