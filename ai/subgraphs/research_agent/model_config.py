from ai.models.gpt import gpt5_mini, gpt5_nano

# Model configuration for graph nodes
MODEL_CONFIG = {
    "create_conversation": gpt5_nano,
    "query_vector_db": gpt5_nano,
    "write_queries": gpt5_nano,
    "assess_resources_classifier": gpt5_nano,
    "assess_resources_feedback": gpt5_nano,
    "write_response": gpt5_mini
}
