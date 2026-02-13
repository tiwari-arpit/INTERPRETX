class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register(self, name, model):
        self.models[name] = model

    def get(self, name):
        if name not in self.models:
            raise ValueError(f"Model '{name}' is not registered")
        return self.models[name]

    def names(self):
        return list(self.models.keys())
