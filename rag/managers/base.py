class BasePipelineManager:
    def __init__(self) -> None:
        self.config = {}
        
    def set_config(self, config: dict):
        raise NotImplementedError()