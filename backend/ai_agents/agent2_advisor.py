import json

class Agent2Advisor:
    def __init__(self, knowledge_base_path):
        self.knowledge = self._load_knowledge(knowledge_base_path)
        
    def _load_knowledge(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading knowledge base: {str(e)}")
            return {}

    def get_advice(self, disease_name):
        return self.knowledge.get(disease_name, "Recommended action: Consult agricultural expert.")
