def clustering_system_prompt(self):
    if self.data_type == "media":
        goal = "understanding if there exists a generalizable framing dimension to influence us with decisions on policy issues."
    elif self.data_type == "astro":
        goal = "understanding the query type to the literature search bot from the astronomy scientists."
    elif self.data_type == "emo":
        goal = "understanding the specific research motivations of the given paper about emotion recognition."
    elif self.data_type == "val_e":
        goal = "identifying the values of this machine learning research."
    else:
        raise ValueError(f"Unknown dataset: {self.data_type}")
    system_prompt = f"""
    Synthesize the entire list of labels by clustering similar labels that are inductively labeled. 
    The clustering is to finalize MEANINGFUL and INSIGHTFUL THEMES for {goal}
    
    You will be provided with an existing codebook. Now you need to cluster the codes into clusters and provide one higher level code for each cluster of codes.

    Guidelines for Clustering:
    - Analyze existing codes and their corresponding segments and cluster the existing codes into clusters with corresponding higher level codes representing the whole cluster.

    The existing codebook will be provided as input following this example below:
    1. <code1>

    -> <segment labeled with code1>

    2. <code2>

    -> <segment labeled with code2>

    ...

    n. <codeN>

    -> <segment labeled with codeN>

    Provide your answers following this output format example below:
    Ans:
    {
    "clusters": [
        {
        "high_level_code": "Cyber Harassment",
        "original_codes": ["Online Harassment", "Cyberbullying"],
        "justification": "Both refer to aggressive online behavior; Cyberbullying is a subset but can be generalized."
        }
    ]
    }

    When no clustering is needed, answer N/A and provide your answer following this output format below:
    Ans: N/A
    """
    return system_prompt