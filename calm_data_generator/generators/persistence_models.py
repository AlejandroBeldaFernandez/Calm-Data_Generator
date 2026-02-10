try:
    import torch
    import torch.nn as nn

    class SimpleDenoiser(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 128),  # +1 for timestep
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, dim),
            )

        def forward(self, x, t):
            # Normalize timestep (assuming steps=1000 roughly or passed in?)
            # In the original code it was t / steps. Steps was local.
            # We can change it to expect t in [0,1] or handle steps dynamic.
            # In original code: t_emb = t.unsqueeze(-1) / steps
            # To be safe, we should pass t normalized or know steps.
            # Let's assume t is raw index and we normalize by 1000 if not provided?
            # Or better, let's just use the value passed.
            # Wait, `steps` was a closure variable in original.
            # We need to pass `steps` to __init__?
            # Or assume t is already normalized when passed?
            # Original: pred_noise = model(noisy, t.float())
            # inside: t_emb = t.unsqueeze(-1) / steps
            # We need to change __init__ to accept steps or method signature.
            # For simplicity let's stick to the previous logic but maybe make steps an attribute?
            t_emb = t.unsqueeze(-1)  # Placeholder, see note
            return self.net(torch.cat([x, t_emb], dim=-1))

    # Redefine properly for picklability
    class SimpleDenoiser(nn.Module):
        def __init__(self, dim, steps=1000):
            super().__init__()
            self.steps = steps
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, dim),
            )

        def forward(self, x, t):
            t_emb = t.unsqueeze(-1) / self.steps
            return self.net(torch.cat([x, t_emb], dim=-1))

except ImportError:

    class SimpleDenoiser:
        pass


try:
    import pandas as pd
    import numpy as np

    class FCSModel:
        """
        A persistent model container for Fully Conditional Specification (FCS) methods.
        Stores conditional models for each feature and marginal distributions for initialization.
        """

        def __init__(self, models, marginals, encoding_info, visit_order):
            self.models = models  # Dict[col, model]
            self.marginals = marginals  # Dict[col, values_to_sample]
            self.encoding_info = (
                encoding_info  # Dict[col, categories] for reconstruction
            )
            self.visit_order = visit_order  # List[col]

        def generate(self, n_samples):
            # 1. Initialize from marginals
            synth_data = {}
            for col, values in self.marginals.items():
                synth_data[col] = np.random.choice(values, size=n_samples, replace=True)

            X_synth = pd.DataFrame(synth_data)

            # Ensure categories use correct dtype if present in encoding_info
            if self.encoding_info:
                for col, categories in self.encoding_info.items():
                    if col in X_synth.columns:
                        X_synth[col] = pd.Categorical(
                            X_synth[col], categories=categories
                        )

            # 2. Iterate once (Gibbs step) to apply conditional models
            for col in self.visit_order:
                if col not in self.models:
                    continue

                model = self.models[col]
                Xs = X_synth.drop(columns=col)

                # Apply encoding if model is not LGBM (heuristic)
                # We need to check if the model object expects encoded inputs.
                # In _synthesize_fcs_generic, we checked "LGBM" in class name.
                is_lgbm = "LGBM" in model.__class__.__name__

                Xs_encoded = Xs.copy()
                if not is_lgbm:
                    for c in Xs_encoded.select_dtypes(include=["category"]).columns:
                        if c in self.encoding_info:
                            # Force categories alignment
                            Xs_encoded[c] = pd.Categorical(
                                Xs_encoded[c], categories=self.encoding_info[c]
                            )
                        # Encode to codes as expected by sklearn trees
                        Xs_encoded[c] = Xs_encoded[c].cat.codes

                # Predict
                new_vals = None

                # Check for probabilistic sampling
                if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
                    try:
                        probs = model.predict_proba(Xs_encoded)
                        classes = model.classes_
                        # Probabilistic sampling
                        # Optimization: If binary, use vectorization
                        if len(classes) == 2:
                            p1 = probs[:, 1]
                            draws = np.random.rand(n_samples)
                            preds_idx = (draws < p1).astype(int)
                            new_vals = classes[preds_idx]
                        else:
                            # Slow row-wise sampling for multiclass
                            new_vals = np.array(
                                [np.random.choice(classes, p=p) for p in probs]
                            )
                    except Exception:
                        # Fallback to direct prediction if proba fails
                        new_vals = model.predict(Xs_encoded)
                else:
                    new_vals = model.predict(Xs_encoded)

                # Restore categorical type if needed
                if col in self.encoding_info:
                    new_vals = pd.Categorical(
                        new_vals, categories=self.encoding_info[col]
                    )

                X_synth[col] = new_vals

            return X_synth

except ImportError:

    class FCSModel:
        pass
