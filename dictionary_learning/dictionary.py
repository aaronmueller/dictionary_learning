"""
Defines the dictionary classes
"""

from abc import ABC, abstractmethod
import torch as t
import torch.nn as nn
import torch.nn.init as init
import einops


class Dictionary(ABC, nn.Module):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """

    dict_size: int  # number of features in the dictionary
    activation_dim: int  # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        pass

    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "Dictionary":
        """
        Load a pretrained dictionary from a file.
        """
        pass


class AutoEncoder(Dictionary, nn.Module):
    """
    A one-layer autoencoder.
    """

    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)

        # initialize encoder and decoder weights
        w = t.randn(activation_dim, dict_size)
        ## normalize columns of w
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        ## set encoder and decoder weights
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

    def encode(self, x):
        return nn.ReLU()(self.encoder(x - self.bias))

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well as the decoded x
        ghost_mask : if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is None:  # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)
            if output_features:
                return x_hat, f
            else:
                return x_hat

        else:  # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)

            x_ghost = self.decoder(
                f_ghost
            )  # note that this only applies the decoder weight matrix, no bias
            x_hat = self.decode(f)
            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.bias.data *= scale

    def normalize_decoder(self):
        norms = t.norm(self.decoder.weight, dim=0).to(dtype=self.decoder.weight.dtype, device=self.decoder.weight.device)

        if t.allclose(norms, t.ones_like(norms)):
            return
        print("Normalizing decoder weights")

        test_input = t.randn(10, self.activation_dim)
        initial_output = self(test_input)

        self.decoder.weight.data /= norms

        new_norms = t.norm(self.decoder.weight, dim=0)
        assert t.allclose(new_norms, t.ones_like(new_norms))

        self.encoder.weight.data *= norms[:, None]
        self.encoder.bias.data *= norms

        new_output = self(test_input)

        # Errors can be relatively large in larger SAEs due to floating point precision
        assert t.allclose(initial_output, new_output, atol=1e-4)


    @classmethod
    def from_pretrained(cls, path, dtype=t.float, device=None, normalize_decoder=True):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = cls(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)

        # This is useful for doing analysis where e.g. feature activation magnitudes are important
        # If training the SAE using the April update, the decoder weights are not normalized
        if normalize_decoder:
            autoencoder.normalize_decoder()

        if device is not None:
            autoencoder.to(dtype=dtype, device=device)

        return autoencoder


class IdentityDict(Dictionary, nn.Module):
    """
    An identity dictionary, i.e. the identity function.
    """

    def __init__(self, activation_dim=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = activation_dim

    def encode(self, x):
        return x

    def decode(self, f):
        return f

    def forward(self, x, output_features=False, ghost_mask=None):
        if output_features:
            return x, x
        else:
            return x

    @classmethod
    def from_pretrained(cls, path, dtype=t.float, device=None):
        """
        Load a pretrained dictionary from a file.
        """
        return cls(None)


class GatedAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with separate gating and magnitude networks.
    """

    def __init__(self, activation_dim, dict_size, initialization="default", device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.decoder_bias = nn.Parameter(t.empty(activation_dim, device=device))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=False, device=device)
        self.r_mag = nn.Parameter(t.empty(dict_size, device=device))
        self.gate_bias = nn.Parameter(t.empty(dict_size, device=device))
        self.mag_bias = nn.Parameter(t.empty(dict_size, device=device))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False, device=device)
        if initialization == "default":
            self._reset_parameters()
        else:
            initialization(self)

    def _reset_parameters(self):
        """
        Default method for initializing GatedSAE weights.
        """
        # biases are initialized to zero
        init.zeros_(self.decoder_bias)
        init.zeros_(self.r_mag)
        init.zeros_(self.gate_bias)
        init.zeros_(self.mag_bias)

        # decoder weights are initialized to random unit vectors
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)
        self.encoder.weight = nn.Parameter(dec_weight.clone().T)

    def encode(self, x: t.Tensor, return_gate:bool=False, normalize_decoder:bool=False):
        """
        Returns features, gate value (pre-Heavyside)
        """
        x_enc = self.encoder(x - self.decoder_bias)

        # gating network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).to(self.encoder.weight.dtype)

        # magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = nn.ReLU()(pi_mag)

        f = f_gate * f_mag

        if normalize_decoder:
            # If the SAE is trained without ConstrainedAdam, the decoder vectors are not normalized
            # Normalizing after encode, and renormalizing before decode to enable comparability
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if return_gate:
            return f, nn.ReLU()(pi_gate)

        return f

    def decode(self, f: t.Tensor, normalize_decoder:bool=False):
        if normalize_decoder:
            # If the SAE is trained without ConstrainedAdam, the decoder vectors are not normalized
            # Normalizing after encode, and renormalizing before decode to enable comparability
            f = f / self.decoder.weight.norm(dim=0, keepdim=True)
        return self.decoder(f) + self.decoder_bias

    def forward(self, x:t.Tensor, output_features:bool=False, normalize_decoder:bool=False):
        f = self.encode(x)
        x_hat = self.decode(f)

        if normalize_decoder:
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if output_features:
            return x_hat, f
        else:
            return x_hat

    def scale_biases(self, scale: float):
        self.decoder_bias.data *= scale
        self.mag_bias.data *= scale
        self.gate_bias.data *= scale

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = GatedAutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class JumpReluAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with jump ReLUs.
    """

    def __init__(self, activation_dim, dict_size, device="cpu"):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.W_enc = nn.Parameter(t.empty(activation_dim, dict_size, device=device))
        self.b_enc = nn.Parameter(t.zeros(dict_size, device=device))
        self.W_dec = nn.Parameter(
            t.nn.init.kaiming_uniform_(t.empty(dict_size, activation_dim, device=device))
        )
        self.b_dec = nn.Parameter(t.zeros(activation_dim, device=device))
        self.threshold = nn.Parameter(t.ones(dict_size, device=device) * 0.001)  # Appendix I

        self.apply_b_dec_to_input = False

        self.W_dec.data = self.W_dec / self.W_dec.norm(dim=1, keepdim=True)
        self.W_enc.data = self.W_dec.data.clone().T

    def encode(self, x, output_pre_jump=False):
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        pre_jump = x @ self.W_enc + self.b_enc

        f = nn.ReLU()(pre_jump * (pre_jump > self.threshold))

        if output_pre_jump:
            return f, pre_jump
        else:
            return f

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features (and their pre-jump version) as well as the decoded x
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        else:
            return x_hat

    def scale_biases(self, scale: float):
        self.b_dec.data *= scale
        self.b_enc.data *= scale
        self.threshold.data *= scale

    @classmethod
    def from_pretrained(
        cls,
        path: str | None = None,
        load_from_sae_lens: bool = False,
        dtype: t.dtype = t.float32,
        device: t.device | None = None,
        **kwargs,
    ):
        """
        Load a pretrained autoencoder from a file.
        If sae_lens=True, then pass **kwargs to sae_lens's
        loading function.
        """
        if not load_from_sae_lens:
            state_dict = t.load(path)
            activation_dim, dict_size = state_dict["W_enc"].shape
            autoencoder = JumpReluAutoEncoder(activation_dim, dict_size)
            autoencoder.load_state_dict(state_dict)
            autoencoder = autoencoder.to(dtype=dtype, device=device)
        else:
            from sae_lens import SAE

            sae, cfg_dict, _ = SAE.from_pretrained(**kwargs)
            assert (
                cfg_dict["finetuning_scaling_factor"] == False
            ), "Finetuning scaling factor not supported"
            dict_size, activation_dim = cfg_dict["d_sae"], cfg_dict["d_in"]
            autoencoder = JumpReluAutoEncoder(activation_dim, dict_size, device=device)
            autoencoder.load_state_dict(sae.state_dict())
            autoencoder.apply_b_dec_to_input = cfg_dict["apply_b_dec_to_input"]

        if device is not None:
            device = autoencoder.W_enc.device
        return autoencoder.to(dtype=dtype, device=device)


# TODO merge this with AutoEncoder
class AutoEncoderNew(Dictionary, nn.Module):
    """
    The autoencoder architecture and initialization used in https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """

    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)

        # initialize encoder and decoder weights
        w = t.randn(activation_dim, dict_size)
        ## normalize columns of w
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        ## set encoder and decoder weights
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

        # initialize biases to zeros
        init.zeros_(self.encoder.bias)
        init.zeros_(self.decoder.bias)

    def encode(self, x):
        return nn.ReLU()(self.encoder(x))

    def decode(self, f):
        return self.decoder(f)

    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        """
        if not output_features:
            return self.decode(self.encode(x))
        else:  # TODO rewrite so that x_hat depends on f
            f = self.encode(x)
            x_hat = self.decode(f)
            # multiply f by decoder column norms
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)
            return x_hat, f

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = AutoEncoderNew(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class RelaxedArchetypalAutoEncoder(Dictionary, nn.Module):
    """
    Autoencoder implementing the Relaxed Archetypal Dictionary for SAE (RA-SAE).

    Constructs a dictionary where each atom is a convex combination of data
    points from an estimated hull, with a small relaxation term constrained by delta.

    For more details, see the paper:
    "Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction
    in Large Vision Models" by Fel et al. (2025), https://arxiv.org/abs/2502.12892

    Parameters
    ----------
    activation_dim : int
        Dimensionality of the input data (e.g number of channels).
    dict_size : int
        Number of components/concepts in the dictionary. The dictionary is overcomplete if
        the number of concepts > activation_dim.
    num_candidates : int
        Number of candidate points to use for the convex hull estimation.
    delta : float, optional
        Constraint on the relaxation term, by default 1.0.
    use_multiplier : bool, optional
        Whether to train a positive scalar to multiply the dictionary after convex combination,
        by default True.
    """

    def __init__(
        self, 
        activation_dim, 
        dict_size, 
        num_candidates=100, 
        delta=1.0, 
        use_multiplier=True,
        device="cpu"
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.delta = delta
        self.num_candidates = num_candidates
        
        # Initialize candidate points randomly - will be updated during training
        self.register_buffer("C", t.randn(num_candidates, activation_dim, device=device))
        
        # Initialize encoder and decoder
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)  # No bias for decoder
        
        # Initialize weights - adapted from AutoEncoderNew
        w = t.randn(activation_dim, dict_size, device=device)
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        
        # Set encoder and decoder weights
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())
        
        # Initialize encoder bias to zeros
        init.zeros_(self.encoder.bias)
        
        # Use a weight matrix for convex combinations of candidates
        self.combination_weights = nn.Parameter(t.zeros(dict_size, num_candidates, device=device))
        
        # Initialize combination weights to be row-stochastic
        with t.no_grad():
            nn.init.kaiming_uniform_(self.combination_weights)
            self.combination_weights.data = nn.functional.softmax(self.combination_weights, dim=-1)
        
        # Relaxation term
        self.Relax = nn.Parameter(t.zeros(dict_size, activation_dim, device=device))
        
        # Multiplier for scaling the dictionary
        if use_multiplier:
            self.multiplier = nn.Parameter(t.tensor(0.0, device=device))
        else:
            self.register_buffer("multiplier", t.tensor(0.0, device=device))
        
        # Dictionary cache
        self._dictionary_cache = None
        
        # Update candidate points with a running estimation
        self.register_buffer("C_update_counter", t.tensor(0, device=device))

    def encode(self, x):
        """
        Encode the input to sparse features.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, activation_dim).
            
        Returns
        -------
        torch.Tensor
            Sparse features of shape (batch_size, dict_size).
        """
        return nn.functional.relu(self.encoder(x))
    
    def decode(self, f):
        """
        Decode the sparse features back to input space.
        
        Parameters
        ----------
        f : torch.Tensor
            Sparse features of shape (batch_size, dict_size).
            
        Returns
        -------
        torch.Tensor
            Decoded tensor of shape (batch_size, activation_dim).
        """
        # Apply decoder transformation
        return self.decoder(f)
    
    def forward(self, x, output_features=False):
        """
        Forward pass of the autoencoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, activation_dim).
        output_features : bool, optional
            Whether to output the sparse features alongside reconstructions, by default False.
            
        Returns
        -------
        torch.Tensor or tuple
            Reconstructed tensor of shape (batch_size, activation_dim) if output_features is False,
            otherwise a tuple (reconstruction, features).
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        
        if not output_features:
            return x_hat
        else:
            # Scale features by decoder column norms for better interpretability
            atom_norms = self.decoder.weight.norm(dim=0, keepdim=True).T.squeeze()
            f_scaled = f * atom_norms
            return x_hat, f_scaled
    
    def _update_dictionary(self):
        """
        Update the decoder weights based on the archetypal dictionary formulation.
        This constrains the dictionary to be a convex combination of candidate points
        plus a relaxation term.
        """
        # Ensure combination_weights remains row-stochastic (positive and row sum to one)
        with t.no_grad():
            W = t.relu(self.combination_weights)
            W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
            self.combination_weights.data = W

            # Enforce the norm constraint on Relax to limit deviation from conv(C)
            norm_Lambda = self.Relax.norm(dim=-1, keepdim=True)  # norm per row
            scaling_factor = t.clamp(self.delta / norm_Lambda, max=1.0)  # safe scaling factor
            self.Relax.data = self.Relax * scaling_factor  # scale Lambda to satisfy ||Lambda|| < delta

        # Compute the dictionary as a convex combination plus relaxation
        D = t.matmul(self.combination_weights, self.C) + self.Relax
        D = D * t.exp(self.multiplier)
        
        # Update the decoder weights directly
        self.decoder.weight.data = D.T
        
        # Cache the updated dictionary
        self._dictionary_cache = D

    def update_candidates(self, batch):
        """
        Update the candidate points using the current batch.
        
        Parameters
        ----------
        batch : torch.Tensor
            Batch of points to use for updating the candidates.
        """
        with t.no_grad():
            # Simple running average update of candidate points
            batch_size = batch.shape[0]
            if batch_size > 0:
                # Select a subset of points if batch is larger than num_candidates
                if batch_size > self.num_candidates:
                    indices = t.randperm(batch_size)[:self.num_candidates]
                    batch = batch[indices]
                
                # Update counter
                self.C_update_counter += 1
                
                # Update C with running average
                alpha = 1.0 / self.C_update_counter
                self.C.data = (1 - alpha) * self.C + alpha * batch[:self.num_candidates]
                
                # Update dictionary since C has changed
                self._update_dictionary()
    
    def apply_archetypal_constraints(self):
        """
        Apply the archetypal constraints to the decoder weights.
        This should be called after each optimization step during training.
        """
        self._update_dictionary()
    
    @classmethod
    def from_pretrained(cls, path, device=None):
        """
        Load a pretrained autoencoder from a file.
        
        Parameters
        ----------
        path : str
            Path to the saved model.
        device : str, optional
            Device to load the model on, by default None.
            
        Returns
        -------
        RelaxedArchetypalAutoEncoder
            Loaded model.
        """
        state_dict = t.load(path)
        
        # Extract parameters from state dict
        C = state_dict.get("C")
        num_candidates = C.shape[0]
        activation_dim = C.shape[1]
        dict_size = state_dict.get("combination_weights").shape[0]
        
        # Create model
        model = cls(activation_dim, dict_size, num_candidates)
        model.load_state_dict(state_dict)
        
        # Update dictionary immediately
        model._update_dictionary()
        
        if device is not None:
            model.to(device)
        
        return model