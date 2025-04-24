"""
AETHER-X: Advanced Self-Optimizing Quantum Trading System
Enhanced with Sophisticated Quantum Circuits, Dynamic Optimization, and Adaptive Market Impact Modeling
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import ray
import time
import subprocess
import json
import sys
import logging
import os
import traceback
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.covariance import MinCovDet
from scipy.stats import genpareto, norm, t
from transformers import pipeline
from tensorflow_probability import distributions as tfd
from scipy.optimize import minimize, differential_evolution
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aether_x.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AETHER-X")

# ---------------------------
# Advanced Quantum Circuit Design
# ---------------------------

class EnhancedQuantumFeatureEngine(tf.keras.Model):
    """Advanced quantum feature extractor with multi-layered parameterized circuits"""
    
    def __init__(self, n_qubits: int = 16, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        
        # More sophisticated parameter structure with layer-specific parameters
        param_shape = [n_layers, 5, n_qubits]  # [layers, gate_types, qubits]
        self.params = tf.Variable(
            tf.random.uniform(param_shape, 0, 2 * np.pi),
            trainable=True,
            name="quantum_circuit_params"
        )
        
        # Create readout operators for multiple observables
        self.z_obs = [cirq.Z(q) for q in self.qubits]
        self.xx_obs = [cirq.PauliString(cirq.X(self.qubits[i]) * cirq.X(self.qubits[(i+1) % n_qubits])) 
                      for i in range(n_qubits)]
        
        # Parameter symbols for TFQ
        self.symbols = sympy.symbols(f'p0:{n_layers*5*n_qubits}')
        self.symbol_values_map = {}
        
    def _prepare_initial_state(self, inputs):
        """Enhanced quantum state preparation with amplitude encoding"""
        # Normalize inputs
        normalized = tf.nn.l2_normalize(inputs, axis=-1)
        
        # Amplitude encoding circuit
        circuit = cirq.Circuit()
        
        # Handle different input dimensions with adaptive encoding
        if inputs.shape[-1] >= self.n_qubits:
            # Subsample or truncate if input dimension is larger than qubit count
            data_for_encoding = normalized[:self.n_qubits]
        else:
            # Pad with zeros if input dimension is smaller than qubit count
            padding = tf.zeros([self.n_qubits - inputs.shape[-1]])
            data_for_encoding = tf.concat([normalized, padding], axis=0)
            
        # Single-qubit rotations based on input data
        for i, q in enumerate(self.qubits):
            circuit.append([
                cirq.ry(np.pi * data_for_encoding[i])(q),
                cirq.rz(np.pi * data_for_encoding[i])(q)
            ])
            
        return circuit
    
    def _entangling_layer(self, qubits, layer_idx: int):
        """Multi-entanglement pattern with various entangling gates"""
        circuit = cirq.Circuit()
        
        # Nearest-neighbor entanglement
        for i in range(0, len(qubits)-1, 2):
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
            
        # Cross-entanglement for even-indexed qubits
        for i in range(0, len(qubits)-2, 4):
            if i+2 < len(qubits):
                circuit.append(cirq.CNOT(qubits[i], qubits[i+2]))
                
        # Periodic boundary for odd layers (creating a cycle)
        if layer_idx % 2 == 1 and len(qubits) > 2:
            circuit.append(cirq.CNOT(qubits[-1], qubits[0]))
            
        return circuit
    
    def _rotation_layer(self, qubits, layer_idx: int):
        """Parameterized rotation layer with multiple rotation axes"""
        circuit = cirq.Circuit()
        param_idx_base = layer_idx * 5 * self.n_qubits
        
        for i, q in enumerate(qubits):
            # Extract the 5 parameters for this qubit in this layer
            param_idx = param_idx_base + i * 5
            symbols = self.symbols[param_idx:param_idx+5]
            
            # Apply different rotation gates
            circuit.append([
                cirq.rx(symbols[0])(q),
                cirq.ry(symbols[1])(q),
                cirq.rz(symbols[2])(q),
                cirq.PhasedXPowGate(phase_exponent=symbols[3], exponent=symbols[4])(q)
            ])
            
        return circuit
    
    def build_circuit(self, inputs):
        """Construct full parameterized quantum circuit"""
        circuit = self._prepare_initial_state(inputs)
        
        # Build multi-layered circuit
        for layer in range(self.n_layers):
            circuit += self._rotation_layer(self.qubits, layer)
            circuit += self._entangling_layer(self.qubits, layer)
            
        # Final rotation layer
        circuit += self._rotation_layer(self.qubits, self.n_layers-1)
        
        return circuit
    
    def _flatten_params(self):
        """Flatten parameters for TFQ compatibility"""
        return tf.reshape(self.params, [-1])
    
    def call(self, inputs):
        """Compute quantum expectation values for multiple observables"""
        flattened_inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])
        batch_size = tf.shape(flattened_inputs)[0]
        
        # Build circuit for the first input to extract symbols
        example_circuit = self.build_circuit(flattened_inputs[0])
        
        # Prepare parameters
        flat_params = self._flatten_params()
        batch_params = tf.tile(tf.expand_dims(flat_params, 0), [batch_size, 1])
        
        # Create batched circuits
        input_circuits = []
        for i in range(batch_size):
            input_circuits.append(self.build_circuit(flattened_inputs[i]))
            
        # Use TFQ to get Z observables expectations
        z_expectations = tfq.layers.Expectation()(
            input_circuits,
            operators=self.z_obs,
            symbol_names=self.symbols,
            symbol_values=batch_params
        )
        
        # Use TFQ to get XX observables expectations 
        xx_expectations = tfq.layers.Expectation()(
            input_circuits,
            operators=self.xx_obs[:len(self.z_obs)//2],  # Using half the number of XX observables
            symbol_names=self.symbols,
            symbol_values=batch_params
        )
        
        # Combine all expectation values
        combined_expectations = tf.concat([z_expectations, xx_expectations], axis=1)
        
        # Apply classical post-processing
        dense = tf.keras.layers.Dense(64, activation='elu')(combined_expectations)
        output = tf.keras.layers.Dense(32, activation='tanh')(dense)
        
        return output

# ---------------------------
# Advanced Market Impact Model
# ---------------------------

@dataclass
class MarketState:
    """Market state representation with microstructure features"""
    volatility: float
    spread: float
    depth: float
    order_imbalance: float
    market_impact_coefficient: float
    temporary_impact_decay: float
    permanent_impact_factor: float

class AdaptiveMarketImpactModel:
    """Sophisticated market impact model with adaptive parameters"""
    
    def __init__(self):
        self.alpha = 0.5  # Permanent impact coefficient
        self.beta = 1.5   # Temporary impact exponent
        self.eta = 1e-6   # Risk aversion parameter
        self.gamma = 0.3  # Market depth parameter
        self.scaler = StandardScaler()
        self.market_regime_detector = MarketRegimeDetector()
        self.impact_history = []
        self.calibration_window = 100
        
    def estimate_market_impact(self, volume: float, volatility: float, 
                              market_depth: float, spread: float) -> tuple:
        """Estimate market impact based on current market conditions"""
        # Square-root law with adaptive coefficients
        permanent_impact = self.alpha * spread * np.power(volume / market_depth, self.gamma)
        
        # Temporary impact with volatility scaling
        temporary_impact = spread * (1 + volatility * 10) * np.power(volume / market_depth, self.beta)
        
        return permanent_impact, temporary_impact
    
    def _almgren_chriss_trajectory(self, X: float, T: float, n: int, 
                                 market_state: MarketState) -> np.ndarray:
        """Compute optimal execution trajectory using advanced Almgren-Chriss model"""
        # Extract market state parameters
        sigma = market_state.volatility
        perm_impact = market_state.permanent_impact_factor
        temp_impact = market_state.market_impact_coefficient
        decay = market_state.temporary_impact_decay
        
        # Enhanced model with decay factor
        k = np.sqrt(self.eta * sigma**2 / temp_impact)
        t = np.linspace(0, T, n)
        
        # Include temporary impact decay in the model
        sinh_kT = np.sinh(k*T)
        cosh_kT = np.cosh(k*T)
        
        # Calculate trajectory with decay-adjusted coefficients
        numerator = np.sinh(k*(T - t)) + decay * k * temp_impact * np.cosh(k*(T - t))
        denominator = sinh_kT + decay * k * temp_impact * cosh_kT
        
        optimal_path = X * numerator / denominator
        
        # Calculate trading rate (derivative of the position trajectory)
        if n > 1:
            trading_rate = np.abs(np.diff(optimal_path, prepend=X))
        else:
            trading_rate = np.array([X])
            
        return optimal_path, trading_rate
    
    def compute_optimal_execution(self, order: Dict, market_state: MarketState) -> Dict:
        """Compute optimal execution schedule with market impact considerations"""
        # Determine the execution horizon based on order size and volatility
        adaptive_horizon = order.get('horizon', 3600) * (1 + 0.5 * market_state.volatility)
        
        # Adjust number of slices based on market depth and volatility
        adaptive_slices = max(10, int(order.get('slices', 10) * 
                                    (1 + market_state.volatility - 0.5 * market_state.depth)))
        
        # Compute the trajectory
        position_traj, trading_rate = self._almgren_chriss_trajectory(
            order['quantity'],
            adaptive_horizon,
            adaptive_slices,
            market_state
        )
        
        # Calculate expected transaction costs
        permanent_impact = market_state.permanent_impact_factor * order['quantity']
        temporary_impact_cost = sum(trading_rate**2) * market_state.market_impact_coefficient
        risk_cost = 0.5 * self.eta * market_state.volatility**2 * sum(position_traj**2 * (adaptive_horizon/adaptive_slices))
        
        # Create execution schedule with timestamps
        current_time = datetime.now()
        time_delta = adaptive_horizon / adaptive_slices
        
        schedule = []
        for i in range(adaptive_slices):
            if i < len(trading_rate):
                schedule.append({
                    'time': current_time + pd.Timedelta(seconds=i*time_delta),
                    'quantity': trading_rate[i],
                    'expected_price_impact': self._calculate_price_impact(trading_rate[i], market_state)
                })
            
        return {
            'schedule': schedule,
            'costs': {
                'permanent_impact': permanent_impact,
                'temporary_impact': temporary_impact_cost,
                'risk_cost': risk_cost,
                'total_cost': permanent_impact + temporary_impact_cost + risk_cost
            },
            'metrics': {
                'participation_rate': order['quantity'] / (market_state.depth * adaptive_horizon),
                'price_impact_estimate': permanent_impact / order['quantity'],
                'execution_horizon': adaptive_horizon,
                'slices': adaptive_slices
            }
        }
    
    def _calculate_price_impact(self, trade_size: float, market_state: MarketState) -> float:
        """Calculate expected price impact for a single trade"""
        return market_state.market_impact_coefficient * np.power(trade_size, self.beta)
    
    def calibrate(self, historical_trades: List[Dict], market_states: List[MarketState]):
        """Calibrate model parameters using Bayesian optimization"""
        if len(historical_trades) < self.calibration_window:
            logger.warning(f"Insufficient data for calibration: {len(historical_trades)} < {self.calibration_window}")
            return
            
        # Extract features for calibration
        features = []
        realized_impacts = []
        
        for trade, state in zip(historical_trades, market_states):
            features.append([
                trade['quantity'], 
                state.volatility,
                state.depth,
                state.spread,
                state.order_imbalance
            ])
            realized_impacts.append(trade['realized_impact'])
            
        # Standardize features
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)
        
        # Define loss function for optimization
        def loss_function(params):
            self.alpha, self.beta, self.eta, self.gamma = params
            total_squared_error = 0
            
            for i, (feature, impact) in enumerate(zip(scaled_features, realized_impacts)):
                quantity, vol, depth, spread, imbalance = self.scaler.inverse_transform([feature])[0]
                predicted_impact, _ = self.estimate_market_impact(quantity, vol, depth, spread)
                squared_error = (predicted_impact - impact)**2
                total_squared_error += squared_error
                
            return total_squared_error / len(features)
            
        # Use differential evolution for global optimization
        bounds = [(0.1, 1.0), (1.0, 2.0), (1e-7, 1e-5), (0.1, 0.6)]
        result = differential_evolution(loss_function, bounds, popsize=20, mutation=(0.5, 1.0), 
                                        recombination=0.7, maxiter=100)
        
        if result.success:
            self.alpha, self.beta, self.eta, self.gamma = result.x
            logger.info(f"Market impact model calibrated: α={self.alpha:.3f}, β={self.beta:.3f}, "
                      f"η={self.eta:.2e}, γ={self.gamma:.3f}")
        else:
            logger.warning(f"Market impact model calibration failed: {result.message}")

# ---------------------------
# Market Regime Detection
# ---------------------------

class MarketRegimeDetector:
    """Hidden Markov Model-based market regime detector"""
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.regime_means = np.zeros((n_regimes, 5))  # 5 features per regime
        self.regime_covs = np.array([np.eye(5) for _ in range(n_regimes)])
        self.transition_matrix = np.ones((n_regimes, n_regimes)) / n_regimes
        self.initial_probs = np.ones(n_regimes) / n_regimes
        self.current_regime = 0
        self.regime_history = []
        self.trained = False
        
    def train(self, features: np.ndarray):
        """Train HMM model using Baum-Welch algorithm"""
        if features.shape[0] < 100:
            logger.warning(f"Insufficient data for regime detection: {features.shape[0]} < 100")
            return False
            
        try:
            from hmmlearn import hmm
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Initialize and train HMM
            model = hmm.GaussianHMM(n_components=self.n_regimes, covariance_type="full",
                                   n_iter=100, random_state=42)
            model.fit(scaled_features)
            
            # Extract trained parameters
            self.regime_means = model.means_
            self.regime_covs = model.covars_
            self.transition_matrix = model.transmat_
            self.initial_probs = model.startprob_
            
            # Decode states
            self.regime_history = model.predict(scaled_features)
            self.current_regime = self.regime_history[-1]
            self.trained = True
            
            logger.info(f"Market regime model trained successfully, identified {self.n_regimes} regimes")
            return True
            
        except ImportError:
            logger.warning("hmmlearn package not available, using fallback regime detection")
            # Fallback to simpler clustering-based regime detection
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
            clusters = kmeans.fit_predict(features)
            
            # Calculate regime parameters
            for i in range(self.n_regimes):
                cluster_data = features[clusters == i]
                if len(cluster_data) > 0:
                    self.regime_means[i] = np.mean(cluster_data, axis=0)
                    self.regime_covs[i] = np.cov(cluster_data, rowvar=False) + 1e-6 * np.eye(features.shape[1])
            
            # Simple transition matrix
            self.transition_matrix = np.ones((self.n_regimes, self.n_regimes)) / self.n_regimes
            
            # Current regime
            self.regime_history = clusters
            self.current_regime = clusters[-1]
            self.trained = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error in regime detection training: {str(e)}")
            return False
    
    def detect_regime(self, features: np.ndarray) -> int:
        """Detect current market regime based on features"""
        if not self.trained:
            return 0
            
        try:
            # Calculate likelihood for each regime
            likelihoods = np.zeros(self.n_regimes)
            
            for i in range(self.n_regimes):
                # Multivariate normal likelihood
                centered = features - self.regime_means[i]
                precision = np.linalg.inv(self.regime_covs[i])
                likelihoods[i] = -0.5 * centered.dot(precision).dot(centered.T)
                
            # Include transition probabilities if we have a current regime
            if len(self.regime_history) > 0:
                for i in range(self.n_regimes):
                    likelihoods[i] += np.log(self.transition_matrix[self.current_regime, i])
            
            # Get most likely regime
            new_regime = np.argmax(likelihoods)
            self.regime_history.append(new_regime)
            self.current_regime = new_regime
            
            return new_regime
            
        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            return self.current_regime

# ---------------------------
# Enhanced Evolutionary Optimization
# ---------------------------

class AdaptiveCMAESOptimizer:
    """Advanced CMA-ES with adaptive step size and restart strategy"""
    
    def __init__(self, dim: int, bounds: List[Tuple[float, float]] = None):
        self.dim = dim
        self.bounds = bounds if bounds else [(0, 1) for _ in range(dim)]
        
        # Strategy parameters
        self.population_size = 4 + int(3 * np.log(dim))
        self.parent_size = self.population_size // 2
        self.mean = np.zeros(dim)
        self.sigma = 0.5
        self.C = np.eye(dim)  # Covariance matrix
        
        # Evolution path variables
        self.ps = np.zeros(dim)  # Evolution path for sigma
        self.pc = np.zeros(dim)  # Evolution path for C
        
        # Strategy constants
        self.cs = (self.population_size + 2) / (self.dim + self.population_size + 5)
        self.c1 = 2 / ((dim + 1.3)**2 + self.population_size)
        self.cmu = min(1 - self.c1, 2 * (self.population_size - 2 + 1/self.population_size) / 
                     ((dim + 2)**2 + self.population_size))
        self.damps = 1 + 2 * max(0, np.sqrt((self.population_size - 1) / (dim + 1)) - 1) + self.cs
        
        # Weights for recombination
        self.weights = np.log(self.population_size + 0.5) - np.log(np.arange(1, self.population_size + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights**2)
        
        # Restart strategy
        self.restarts = 0
        self.max_restarts = 3
        self.stagnation_counter = 0
        self.stagnation_threshold = 30
        self.best_fitness = float('inf')
        self.fitness_history = []
        
        # Adaptive sampling
        self.sample_size = self.population_size
        self.generation = 0
        
    def _check_bounds(self, x: np.ndarray) -> np.ndarray:
        """Apply bounds to solution vectors"""
        return np.clip(x, 
                      [lower for lower, _ in self.bounds], 
                      [upper for _, upper in self.bounds])
    
    def ask(self) -> np.ndarray:
        """Generate new candidate solutions with adaptive population size"""
        # Increase sample size if optimization is stagnating
        if self.stagnation_counter > self.stagnation_threshold // 2:
            self.sample_size = int(self.population_size * 1.5)
        else:
            self.sample_size = self.population_size
            
        # Eigendecomposition of C
        try:
            D, B = np.linalg.eigh(self.C)
            D = np.sqrt(np.maximum(D, 1e-8))  # Ensure positive eigenvalues
        except np.linalg.LinAlgError:
            # Handle numerical issues by regularizing C
            logger.warning("Eigendecomposition failed, regularizing covariance matrix")
            self.C = self.C + 1e-6 * np.eye(self.dim)
            D, B = np.linalg.eigh(self.C)
            D = np.sqrt(np.maximum(D, 1e-8))
            
        # Generate samples
        z = np.random.randn(self.sample_size, self.dim)
        y = z @ (B * D)
        x = self.mean + self.sigma * y
        
        # Apply bounds
        x = np.array([self._check_bounds(xi) for xi in x])
        
        self.current_z = z  # Save for tell method
        self.current_y = y
        
        return x
    
    def tell(self, x: np.ndarray, fitnesses: np.ndarray):
        """Update distribution parameters based on fitness evaluation"""
        # Sort by fitness
        idx = np.argsort(fitnesses)
        x_sorted = x[idx]
        y_sorted = self.current_y[idx]
        z_sorted = self.current_z[idx]
        
        # Check for improvement
        best_current_fitness = fitnesses[idx[0]]
        self.fitness_history.append(best_current_fitness)
        
        if best_current_fitness < self.best_fitness:
            self.best_fitness = best_current_fitness
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            
        # Check restart condition
        if self.stagnation_counter >= self.stagnation_threshold:
            if self.restarts < self.max_restarts:
                self._restart()
                return
                
        # Use only the best individuals for update
        selected_y = y_sorted[:self.population_size]
        selected_z = z_sorted[:self.population_size]
        
        # Weighted recombination of mean
        old_mean = self.mean.copy()
        self.mean = old_mean + self.sigma * np.sum(self.weights[:, np.newaxis] * selected_y[:len(self.weights)], axis=0)
        
        # Update evolution paths
        y_w = np.sum(self.weights[:, np.newaxis] * selected_y[:len(self.weights)], axis=0)
        
        # Update sigma evolution path
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * \
                 (B @ (D**-1) @ (self.mean - old_mean)) / self.sigma
                 
        # Update covariance evolution path
        hsig = float(np.linalg.norm(self.ps) / 
                   np.sqrt(1 - (1 - self.cs)**(2 * (self.generation + 1))) < 
                   (1.4 + 2 / (self.dim + 1)) * np.sqrt(self.dim))
                   
        self.pc = (1 - self.c1) * self.pc + hsig * np.sqrt(self.c1 * (2 - self.c1) * self.mueff) * \
                 (self.mean - old_mean) / self.sigma
                 
        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)
        
        # Rank-mu update
        rank_mu = np.zeros_like(self.C)
        for i, weight in enumerate(self.weights):
            if i < len(selected_y):
                rank_mu += weight * np.outer(selected_y[i], selected_y[i])
                
        # Update covariance matrix
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * rank_one + self.cmu * rank_mu
        
        # Update step size sigma using cumulative step length adaptation
        self.sigma *= np.exp((np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1) * self.cs / self.damps)
        
        # Bound sigma to avoid numerical issues
        self.sigma = np.clip(self.sigma, 1e-8, 1e2)
        
        self.generation += 1
        
    def _restart(self):
        """Implement restart strategy"""
        logger.info(f"CMA-ES restart #{self.restarts+1}: sigma={self.sigma:.6f}, fitness={self.best_fitness:.6f}")
        
        # Increase step size and randomize mean
        self.sigma = 2.0
        self.mean = np.random.uniform(
            [lower for lower, _ in self.bounds],
            [upper for _, upper in self.bounds]
        )
        
        # Reset evolution paths
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        # Reset covariance matrix but maintain some information
        self.C = np.eye(self.dim)
        
        # Reset counters
        self.stagnation_counter = 0
        self.restarts += 1
        
    def get_best(self) -> Dict:
        """Return best solution found so far"""
        return {
            'solution': self.mean,
            'fitness': self.best_fitness,
            'sigma': self.sigma,
            'generations': self.generation,
            'restarts': self.restarts
        }

# ---------------------------
# Advanced Anomaly Detection
# ---------------------------

class MultivariateExtremeTailDetector:
    """Advanced anomaly detection with multivariate extreme value theory"""
    
    def __init__(self, contamination: float = 0.05, n_components: int = 3):
        self.contamination = contamination
        self.n_components = n_components
        self.mcd = MinCovDet(support_fraction=0.8)
        self.gpd_models = []
        self.threshold = None
        self.mixture_weights = np.ones(n_components) / n_components
        self.feature_scaler = StandardScaler()
        
    def _compute_mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances with robust covariance"""
        return self.mcd.mahalanobis(X)
    
    def _fit_gpd(self, distances: np.ndarray) -> List:
        """Fit Generalized Pareto Distribution to tail"""
        # Determine threshold as percentile
        threshold = np.percentile(distances, 100 * (1 - self.contamination))
        exceedances = distances[distances > threshold] - threshold
        
        if len(exceedances) < 10:
            logger.warning(f"Too few exceedances ({len(exceedances)}) for GPD fitting")
            return [(0.1, 0.1, threshold)]  # Default parameters
            
        # Fit multiple GPD models with different initial parameters
        models = []
        
        for _ in range(self.n_components):
            try:
                shape, scale, _ = genpareto.fit(exceedances, floc=0)
                models.append((shape, scale, threshold))
            except RuntimeError:
                # Handle fitting failures
                models.append((0.1, np.std(exceedances), threshold))
                
        return models
    
    def fit(self, X: np.ndarray):
        """Fit anomaly detection model to data"""
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Fit robust covariance estimator
        self.mcd.fit(X_scaled)
        
        # Compute Mahalanobis distances
        distances = self._compute_mahalanobis(X_scaled)
        
        # Fit mixture of GPD to the tail
        self.gpd_models = self._fit_gpd(distances)
        
        # Determine decision threshold based on contamination
        self.threshold = self._compute_threshold(self.contamination)
        
        # Optimize mixture weights using EM-like algorithm
        self._optimize_mixture_weights(distances)
        
        return self
    
    def _optimize_mixture_weights(self, distances: np.ndarray):
        """Optimize mixture weights using expectation-maximization"""
        tail_samples = distances[distances > self.gpd_models[0][2]]
        
        if len(tail_samples) < self.n_components:
            # Not enough samples for optimization
            self.mixture_weights = np.ones(len(self.gpd_models)) / len(self.gpd_models)
            return
            
        # EM algorithm for 10 iterations
        for _ in range(10):
            # E-step: Compute responsibilities
            responsibilities = np.zeros((len(tail_samples), len(self.gpd_models)))
            
            for i, (shape, scale, threshold) in enumerate(self.gpd_models):
                exceedances = tail_samples - threshold
                pdf_values = genpareto.pdf(exceedances, shape, loc=0, scale=scale)
                responsibilities[:, i] = self.mixture_weights[i] * pdf_values
                
            # Normalize responsibilities
            row_sums = responsibilities.sum(axis=1, keepdims=True)
            responsibilities = responsibilities / np.maximum(row_sums, 1e-10)
            
            # M-step: Update weights
            self.mixture_weights = responsibilities.mean(axis=0)
            self.mixture_weights = self.mixture_weights / self.mixture_weights.sum()
    
    def _compute_threshold(self, target_contamination: float) -> float:
        """Compute decision threshold for specified contamination level"""
        # Set initial threshold as the minimum GPD threshold
        initial_threshold = min([threshold for _, _, threshold in self.gpd_models])
        
        # Define inverse CDF function for mixture model
        def mixture_quantile(p):
            if p >= 1.0:
                return float('inf')
                
            # Convert to exceedance probability
            p_exceed = 1 - p
            
            # Weighted sum of GPD quantiles
            quantile = 0
            for i, (shape, scale, threshold) in enumerate(self.gpd_models):
                weight = self.mixture_weights[i]
                component_quantile = threshold + scale / shape * ((1 - p_exceed) ** (-shape) - 1) if shape != 0 else \
                                     threshold - scale * np.log(p_exceed)
                quantile += weight * component_quantile
                
            return quantile
            
        # Binary search for threshold giving desired contamination
        target_p = 1 - target_contamination
        lower = initial_threshold
        upper = initial_threshold * 100  # Arbitrary large value
        
        for _ in range(20):  # 20 iterations should be enough for convergence
            mid = (lower + upper) / 2
            p_mid = self._score_to_probability(mid)
            
            if abs(p_mid - target_p) < 1e-4:
                break
                
            if p_mid < target_p:
                lower = mid
            else:
                upper = mid
                
        return (lower + upper) / 2
    
    def _score_to_probability(self, score: float) -> float:
        """Convert anomaly score to probability using fitted GPD models"""
        # Compute exceedance probability for each component
        exceedance_probs = []
        
        for shape, scale, threshold in self.gpd_models:
            if score <= threshold:
                exceedance_probs.append(1.0)  # Not in the tail
            else:
                exceedance = score - threshold
                if shape == 0:  # Exponential case
                    p = np.exp(-exceedance / scale)
                else:
                    p = (1 + shape * exceedance / scale) ** (-1 / shape)
                exceedance_probs.append(max(0, min(1, p)))
                
        # Weighted average of component probabilities
        return np.sum(self.mixture_weights * np.array(exceedance_probs))
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores using Mahalanobis distance and GPD mixture"""
        X_scaled = self.feature_scaler.transform(X)
        distances = self._compute_mahalanobis(X_scaled)
        
        # Transform distances using fitted GPD models
        transformed_scores = np.zeros_like(distances)
        
        for i, distance in enumerate(distances):
            # Compute tail probability for each point
            p = self._score_to_probability(distance)
            # Convert to anomaly score (-log of exceedance probability)
            transformed_scores[i] = -np.log(max(p, 1e-10))
            
        return transformed_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict if points are anomalies"""
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)
    
    def anomaly_probability(self, X: np.ndarray) -> np.ndarray:
        """Compute probability that points are anomalies"""
        scores = self.decision_function(X)
        probs = 1 - np.exp(-scores)
        return probs

# ---------------------------
# Signal Processing & Feature Engineering
# ---------------------------

class AdvancedFeatureExtractor:
    """Advanced market feature extraction with signal processing techniques"""
    
    def __init__(self, n_features: int = 50):
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_importance = None
        
    def extract_time_domain_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract time domain features from price data"""
        features = pd.DataFrame(index=price_data.index[20:])  # Skip first 20 rows for indicators
        
        # Price based features
        features['returns'] = price_data['close'].pct_change()
        features['log_returns'] = np.log(price_data['close']).diff()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'ma_{window}'] = price_data['close'].rolling(window=window).mean()
            features[f'std_{window}'] = price_data['close'].rolling(window=window).std()
            
        # Price momentum and acceleration
        features['momentum_5'] = features['returns'].rolling(window=5).sum()
        features['momentum_10'] = features['returns'].rolling(window=10).sum()
        features['acceleration'] = features['returns'].diff()
        
        # Technical indicators
        features['rsi_14'] = self._calculate_rsi(price_data['close'], window=14)
        
        # MACD
        macd, signal, hist = self._calculate_macd(price_data['close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        
        # Volatility measures
        features['atr_14'] = self._calculate_atr(price_data, window=14)
        
        # Volume-based features
        if 'volume' in price_data.columns:
            features['volume_change'] = price_data['volume'].pct_change()
            features['volume_ma_5'] = price_data['volume'].rolling(window=5).mean()
            features['volume_ma_10'] = price_data['volume'].rolling(window=10).mean()
            features['volume_ma_ratio'] = features['volume_ma_5'] / features['volume_ma_10']
            features['price_volume_corr'] = features['returns'].rolling(10).corr(features['volume_change'])
            
        # Mean reversion features
        features['dist_from_ma_20'] = (price_data['close'] / features['ma_20'] - 1)
        features['dist_from_ma_50'] = (price_data['close'] / features['ma_50'] - 1)
        
        return features.dropna()
    
    def extract_frequency_domain_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract frequency domain features using FFT and wavelets"""
        try:
            import pywt
            from scipy.fftpack import fft
            
            # Prepare log returns
            log_returns = np.log(price_data['close']).diff().dropna().values
            
            if len(log_returns) < 128:
                logger.warning(f"Insufficient data for frequency analysis: {len(log_returns)} < 128")
                return pd.DataFrame(index=price_data.index[1:])
                
            # Frequency features DataFrame
            freq_features = pd.DataFrame(index=price_data.index[128:])
            
            # Compute sliding window FFT features
            for i in range(128, len(log_returns)):
                window = log_returns[i-128:i]
                fft_values = fft(window)
                fft_abs = np.abs(fft_values[:64])  # First half of spectrum (real signals are symmetric)
                
                # Extract key frequency components
                freq_features.loc[price_data.index[i], 'fft_max'] = np.max(fft_abs)
                freq_features.loc[price_data.index[i], 'fft_mean'] = np.mean(fft_abs)
                freq_features.loc[price_data.index[i], 'fft_std'] = np.std(fft_abs)
                freq_features.loc[price_data.index[i], 'fft_skew'] = np.mean((fft_abs - np.mean(fft_abs))**3) / (np.std(fft_abs)**3)
                
                # Frequency bands
                freq_features.loc[price_data.index[i], 'fft_low_band'] = np.sum(fft_abs[1:8])
                freq_features.loc[price_data.index[i], 'fft_mid_band'] = np.sum(fft_abs[8:32])
                freq_features.loc[price_data.index[i], 'fft_high_band'] = np.sum(fft_abs[32:64])
                
            # Wavelet decomposition - compute sliding window wavelet features
            for i in range(128, len(log_returns)):
                window = log_returns[i-128:i]
                coeffs = pywt.wavedec(window, 'db4', level=4)
                
                # Extract wavelet features from each decomposition level
                for j, c in enumerate(coeffs):
                    freq_features.loc[price_data.index[i], f'wavelet_{j}_mean'] = np.mean(np.abs(c))
                    freq_features.loc[price_data.index[i], f'wavelet_{j}_std'] = np.std(c)
                    freq_features.loc[price_data.index[i], f'wavelet_{j}_energy'] = np.sum(c**2)
                    
            return freq_features
            
        except ImportError:
            logger.warning("pywt package not available, skipping wavelet features")
            return pd.DataFrame(index=price_data.index[20:])
        except Exception as e:
            logger.error(f"Error in frequency domain feature extraction: {str(e)}")
            return pd.DataFrame(index=price_data.index[20:])
    
    def extract_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract combined feature set from price data"""
        # Extract both time and frequency domain features
        time_features = self.extract_time_domain_features(price_data)
        freq_features = self.extract_frequency_domain_features(price_data)
        
        # Combine features on common index
        common_index = time_features.index.intersection(freq_features.index)
        combined_features = pd.concat([
            time_features.loc[common_index], 
            freq_features.loc[common_index]
        ], axis=1)
        
        # Fill any remaining NAs with forward fill then backward fill
        combined_features = combined_features.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        scaled_features = pd.DataFrame(
            self.scaler.fit_transform(combined_features),
            index=combined_features.index,
            columns=combined_features.columns
        )
        
        return scaled_features
    
    def select_features(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select most important features using machine learning"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.feature_selection import SelectFromModel
            
            # Train a random forest to get feature importances
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(features, target)
            
            # Select features using model-based selection
            selector = SelectFromModel(
                rf, 
                max_features=min(self.n_features, features.shape[1]),
                threshold=-np.inf  # Keep max_features features
            )
            selector.fit(features, target)
            
            # Get selected feature indices
            selected_indices = selector.get_support()
            selected_columns = features.columns[selected_indices]
            
            # Save feature importances
            self.feature_importance = pd.Series(
                rf.feature_importances_[selected_indices],
                index=selected_columns
            ).sort_values(ascending=False)
            
            self.selected_features = selected_columns.tolist()
            
            # Return selected feature matrix
            return features[self.selected_features]
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            # Fall back to using all features
            self.selected_features = features.columns.tolist()
            return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        """Calculate MACD, signal line, and histogram"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        return macd, macd_signal, macd_hist
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr

# ---------------------------
# Neural Network Architecture
# ---------------------------

class QuantumEnhancedNeuralNetwork(tf.keras.Model):
    """Neural network with quantum feature extraction layer"""
    
    def __init__(self, n_qubits: int = 16, n_classical_inputs: int = 50, n_outputs: int = 5):
        super().__init__()
        
        # Quantum feature extractor
        self.quantum_layer = EnhancedQuantumFeatureEngine(n_qubits=n_qubits, n_layers=3)
        
        # Classical preprocessing
        self.classical_preprocessor = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='swish'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2)
        ])
        
        # Hybrid feature fusion
        self.fusion_layer = tf.keras.layers.Dense(64, activation='swish')
        
        # Time-aware attention mechanism
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=16
        )
        self.attention_norm = tf.keras.layers.LayerNormalization()
        
        # Output layers with uncertainty estimation
        self.mu = tf.keras.layers.Dense(n_outputs)  # Mean prediction
        self.sigma = tf.keras.layers.Dense(n_outputs, activation='softplus')  # Uncertainty
        
    def call(self, inputs, training=False):
        # Unpack inputs - classical and quantum inputs
        classical_inputs, quantum_inputs = inputs
        
        # Process classical features
        classical_features = self.classical_preprocessor(classical_inputs, training=training)
        
        # Process quantum features
        quantum_features = self.quantum_layer(quantum_inputs)
        
        # Concatenate classical and quantum features
        combined_features = tf.concat([classical_features, quantum_features], axis=-1)
        
        # Apply feature fusion
        fused_features = self.fusion_layer(combined_features)
        
        # Apply self-attention for temporal dependencies (reshape for sequence)
        batch_size = tf.shape(fused_features)[0]
        reshaped_features = tf.reshape(fused_features, [batch_size, -1, 64])
        
        # Self-attention mechanism
        attended_features = self.attention(
            reshaped_features, reshaped_features, 
            attention_mask=None, training=training
        )
        attended_features = self.attention_norm(attended_features + reshaped_features)
        
        # Flatten for output layers
        flattened = tf.reshape(attended_features, [batch_size, -1])
        
        # Predict mean and uncertainty
        mu = self.mu(flattened)
        sigma = self.sigma(flattened) + 1e-6  # Add small constant for numerical stability
        
        # Return both mean and uncertainty estimates
        return mu, sigma

# ---------------------------
# Trading Strategy Implementation
# ---------------------------

class QuantumEnhancedStrategy:
    """Main trading strategy with quantum-enhanced decision making"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.feature_extractor = AdvancedFeatureExtractor()
        self.anomaly_detector = MultivariateExtremeTailDetector()
        self.market_impact_model = AdaptiveMarketImpactModel()
        self.regime_detector = MarketRegimeDetector()
        self.model = None
        self.optimizer = None
        self.trained = False
        self.position = 0
        self.last_trade_price = None
        self.performance_metrics = {}
        
        # Strategy parameters (with defaults)
        self.lookback_window = self.config.get('lookback_window', 100)
        self.prediction_horizon = self.config.get('prediction_horizon', 5)
        self.risk_limit = self.config.get('risk_limit', 0.02)
        self.max_position = self.config.get('max_position', 1.0)
        self.transaction_cost = self.config.get('transaction_cost', 0.0005)
        
        # Initialize ray for distributed processing if not initialized
        if not ray.is_initialized():
            try:
                ray.init(ignore_reinit_error=True)
            except Exception as e:
                logger.warning(f"Failed to initialize Ray: {str(e)}")
    
    def _create_model(self, input_dim: int):
        """Create and compile the prediction model"""
        # Set up quantum-enhanced neural network
        n_qubits = min(16, input_dim // 2)  # Limit number of qubits
        self.model = QuantumEnhancedNeuralNetwork(
            n_qubits=n_qubits,
            n_classical_inputs=input_dim,
            n_outputs=self.prediction_horizon
        )
        
        # Compile with custom negative log likelihood loss
        def gaussian_nll_loss(y_true, y_pred):
            mu, sigma = y_pred
            # Negative log likelihood of normal distribution
            return tf.reduce_mean(
                0.5 * tf.math.log(2 * np.pi * sigma**2) + 
                0.5 * ((y_true - mu) / sigma)**2
            )
        
        # Mixed precision optimizer for better performance
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        if tf.config.list_physical_devices('GPU'):
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            
        self.model.compile(optimizer=optimizer, loss=gaussian_nll_loss)
    
    def preprocess_data(self, price_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Process raw price data into features and targets"""
        # Extract features
        features = self.feature_extractor.extract_features(price_data)
        
        # Create prediction targets (future returns)
        targets = pd.DataFrame(index=features.index)
        for i in range(1, self.prediction_horizon + 1):
            targets[f'target_t{i}'] = price_data['close'].pct_change(i).shift(-i).loc[features.index]
            
        # Drop rows with NaN targets
        valid_idx = targets.dropna().index
        features = features.loc[valid_idx]
        targets = targets.loc[valid_idx]
        
        # Create rolling windows of data
        X, y = [], []
        for i in range(self.lookback_window, len(features)):
            feature_window = features.iloc[i-self.lookback_window:i].values
            target = targets.iloc[i-1].values
            
            X.append(feature_window)
            y.append(target)
            
        if not X:
            raise ValueError("Insufficient data for preprocessing")
            
        return np.array(X), np.array(y)
    
    def split_quantum_classical(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split features into quantum and classical parts"""
        # Use the most recent timeframe for quantum features
        quantum_features = X[:, -1, :16]  # Last timeframe, first 16 features
        
        # Use the full sequence for classical features
        classical_features = X.reshape(X.shape[0], -1)
        
        return classical_features, quantum_features
    
    def fit(self, price_data: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """Train the trading model on historical data"""
        try:
            # Preprocess data
            X, y = self.preprocess_data(price_data)
            logger.info(f"Preprocessed data: X shape {X.shape}, y shape {y.shape}")
            
            # Train anomaly detector
            flat_X = X.reshape(X.shape[0], -1)
            self.anomaly_detector.fit(flat_X)
            
            # Train market regime detector
            regime_features = X[:, -1, :5]  # Use last timeframe, first 5 features
            self.regime_detector.train(regime_features)
            
            # Create train/validation split
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Split into quantum and classical features
            X_classical_train, X_quantum_train = self.split_quantum_classical(X_train)
            X_classical_val, X_quantum_val = self.split_quantum_classical(X_val)
            
            # Create model
            self._create_model(X_classical_train.shape[1])
            
            # Setup callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5
                )
            ]
            
            # Train model
            history = self.model.fit(
                [X_classical_train, X_quantum_train],
                y_train,
                epochs=100,
                batch_size=32,
                validation_data=([X_classical_val, X_quantum_val], y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate on validation set
            val_loss = history.history['val_loss'][-1]
            
            # Store performance metrics
            self.performance_metrics = {
                'train_loss': history.history['loss'][-1],
                'val_loss': val_loss,
                'training_samples': len(X_train),
                'val_samples': len(X_val)
            }
            
            self.trained = True
            logger.info(f"Model training completed: validation loss = {val_loss:.6f}")
            
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            traceback.print_exc()
            return {'error': str(e)}
    
    def predict(self, market_data: pd.DataFrame) -> Dict:
        """Generate trading predictions based on current market data"""
        if not self.trained:
            return {'error': 'Model not trained'}
            
        try:
            # Preprocess latest data
            X, _ = self.preprocess_data(market_data)
            
            if len(X) == 0:
                return {'error': 'Insufficient data for prediction'}
                
            # Use the most recent data point
            X_latest = X[-1:] 
            
            # Split into quantum and classical features
            X_classical, X_quantum = self.split_quantum_classical(X_latest)
            
            # Generate prediction with uncertainty
            mu, sigma = self.model.predict([X_classical, X_quantum], verbose=0)
            
            # Get current market regime
            regime_features = X_latest[:, -1, :5]
            current_regime = self.regime_detector.detect_regime(regime_features)
            
            # Check for anomalies
            flat_X = X_latest.reshape(X_latest.shape[0], -1)
            anomaly_score = self.anomaly_detector.decision_function(flat_X)[0]
            is_anomaly = anomaly_score > self.anomaly_detector.threshold
            
            # Extract current market context
            current_price = market_data['close'].iloc[-1]
            current_volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1]
            
            # Prepare prediction results
            predictions = {
                'timestamp': market_data.index[-1],
                'current_price': current_price,
                'predicted_returns': mu[0].tolist(),
                'uncertainty': sigma[0].tolist(),
                'sharpe_ratio': np.mean(mu[0]) / (np.mean(sigma[0]) + 1e-9),
                'regime': int(current_regime),
                'anomaly_score': float(anomaly_score),
                'is_anomaly': bool(is_anomaly),
                'volatility': float(current_volatility)
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {'error': str(e)}
    
    def generate_trade_signal(self, predictions: Dict, current_position: float = 0) -> Dict:
        """Generate trading signal based on predictions"""
        if 'error' in predictions:
            return {'action': 'hold', 'reason': f"Error: {predictions['error']}"}
            
        try:
            # Extract prediction metrics
            returns = predictions['predicted_returns']
            uncertainty = predictions['uncertainty']
            is_anomaly = predictions['is_anomaly']
            current_price = predictions['current_price']
            
            # Calculate risk-adjusted expected return
            expected_return = np.mean(returns)
            prediction_risk = np.mean(uncertainty)
            sharpe = expected_return / (prediction_risk + 1e-9)
            
            # Scale position size by Sharpe ratio and confidence
            confidence = 1.0 / (1.0 + np.mean(uncertainty))
            target_position = np.clip(
                sharpe * confidence * self.max_position,
                -self.max_position,
                self.max_position
            )
            
            # Apply risk limit - reduce position under high volatility or anomalies
            if is_anomaly:
                target_position *= 0.5  # Reduce position size during anomalies
                logger.info(f"Reducing position due to market anomaly (score: {predictions['anomaly_score']:.4f})")
                
            # Adjust for current regime
            regime_factor = 1.0
            if predictions['regime'] == 1:  # Medium volatility regime
                regime_factor = 0.8
            elif predictions['regime'] == 2:  # High volatility regime
                regime_factor = 0.6
                
            target_position *= regime_factor
            
            # Calculate position delta
            position_delta = target_position - current_position
            
            # Determine action
            if abs(position_delta) < 0.1:  # Small change threshold
                action = 'hold'
                quantity = 0
            elif position_delta > 0:
                action = 'buy'
                quantity = position_delta
            else:
                action = 'sell'
                quantity = -position_delta
                
            # Calculate expected impact using market impact model
            market_state = MarketState(
                volatility=predictions['volatility'],
                spread=current_price * 0.0005,  # Estimated spread, replace with actual
                depth=current_price * 1000,    # Estimated depth, replace with actual
                order_imbalance=0.0,           # Unknown, default to balanced
                market_impact_coefficient=1e-6,
                temporary_impact_decay=0.1,
                permanent_impact_factor=1e-7
            )
            
            perm_impact, temp_impact = self.market_impact_model.estimate_impact(
                quantity, 
                current_price,
                market_state
            )
            
            # Adjust execution price based on expected impact
            expected_execution_price = current_price * (1 + temp_impact * np.sign(quantity))
            
            # Calculate expected transaction cost
            transaction_cost = abs(quantity) * current_price * self.transaction_cost
            
            # Generate final signal with all metadata
            signal = {
                'action': action,
                'target_position': float(target_position),
                'current_position': float(current_position),
                'position_delta': float(position_delta),
                'quantity': float(quantity),
                'expected_return': float(expected_return),
                'prediction_risk': float(prediction_risk),
                'sharpe_ratio': float(sharpe),
                'confidence': float(confidence),
                'current_price': float(current_price),
                'expected_execution_price': float(expected_execution_price),
                'estimated_impact': float(perm_impact + temp_impact),
                'transaction_cost': float(transaction_cost),
                'regime': int(predictions['regime']),
                'is_anomaly': bool(is_anomaly),
                'timestamp': predictions['timestamp']
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trade signal: {str(e)}")
            return {'action': 'hold', 'reason': f"Error: {str(e)}"}

    def execute_trade(self, signal: Dict, market_data: pd.DataFrame) -> Dict:
        """Execute the trade based on signal and return execution details"""
        if signal['action'] == 'hold':
            return {'status': 'skipped', 'reason': signal.get('reason', 'No significant position change')}
            
        try:
            # Get current price from market data
            current_price = market_data['close'].iloc[-1]
            quantity = signal['quantity']
            
            # Simulate trade execution with market impact
            execution_price = signal['expected_execution_price']
            transaction_cost = signal['transaction_cost']
            
            # Update position
            if signal['action'] == 'buy':
                self.position += quantity
                self.last_trade_price = execution_price
            elif signal['action'] == 'sell':
                self.position -= quantity
                self.last_trade_price = execution_price
            
            # Calculate P&L impact
            trade_value = quantity * execution_price
            trade_cost = transaction_cost
            
            # Return execution details
            execution = {
                'status': 'executed',
                'action': signal['action'],
                'quantity': float(quantity),
                'price': float(execution_price),
                'value': float(trade_value),
                'cost': float(trade_cost),
                'timestamp': signal['timestamp'],
                'new_position': float(self.position)
            }
            
            logger.info(f"Executed trade: {signal['action']} {quantity:.4f} @ {execution_price:.4f}")
            return execution
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return {'status': 'error', 'reason': str(e)}

    def backtest(self, price_data: pd.DataFrame, train_ratio: float = 0.6) -> Dict:
        """Run backtest on historical data"""
        try:
            # Split data into train and test
            split_idx = int(len(price_data) * train_ratio)
            train_data = price_data.iloc[:split_idx].copy()
            test_data = price_data.iloc[split_idx:].copy()
            
            # Train model on training data
            logger.info(f"Training model on {len(train_data)} samples")
            training_result = self.fit(train_data)
            
            if 'error' in training_result:
                return {'status': 'error', 'reason': training_result['error']}
            
            # Initialize backtest variables
            position = 0.0
            cash = 100000.0  # Starting capital
            holdings = 0.0
            trades = []
            equity_curve = []
            positions = []
            
            # Minimum window size needed for prediction
            min_window = self.lookback_window + self.prediction_horizon + 10
            
            # Process test data in sequential chunks
            for i in range(min_window, len(test_data), 5):  # Step by 5 days
                # Get data window for prediction
                window_end = i
                window_start = max(0, window_end - 300)  # Use at most 300 days
                prediction_window = test_data.iloc[window_start:window_end].copy()
                
                # Skip if not enough data
                if len(prediction_window) < min_window:
                    continue
                    
                # Generate prediction
                current_price = prediction_window['close'].iloc[-1]
                predictions = self.predict(prediction_window)
                
                if 'error' in predictions:
                    logger.warning(f"Skipping prediction: {predictions['error']}")
                    continue
                    
                # Generate trading signal
                signal = self.generate_trade_signal(predictions, position)
                
                # Execute trade
                if signal['action'] != 'hold':
                    execution = self.execute_trade(signal, prediction_window)
                    position = self.position
                    
                    # Update cash and holdings
                    if execution['status'] == 'executed':
                        trade_value = execution['value']
                        trade_cost = execution['cost']
                        
                        if signal['action'] == 'buy':
                            cash -= trade_value
                            cash -= trade_cost
                            holdings += signal['quantity']
                        else:  # sell
                            cash += trade_value
                            cash -= trade_cost
                            holdings -= signal['quantity']
                            
                        trades.append(execution)
                
                # Calculate equity
                portfolio_value = cash + holdings * current_price
                equity_curve.append({
                    'timestamp': test_data.index[window_end-1],
                    'equity': portfolio_value,
                    'cash': cash,
                    'holdings': holdings,
                    'position': position,
                    'price': current_price
                })
                
                positions.append({
                    'timestamp': test_data.index[window_end-1],
                    'position': position
                })
            
            # Convert equity curve to DataFrame
            equity_df = pd.DataFrame(equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Calculate performance metrics
            initial_equity = equity_df['equity'].iloc[0]
            final_equity = equity_df['equity'].iloc[-1]
            returns = equity_df['equity'].pct_change().dropna()
            
            performance = {
                'initial_equity': float(initial_equity),
                'final_equity': float(final_equity),
                'total_return': float((final_equity / initial_equity) - 1),
                'annualized_return': float(((final_equity / initial_equity) ** (252 / len(returns)) - 1)),
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 0 else 0,
                'max_drawdown': float(self._calculate_max_drawdown(equity_df['equity'])),
                'volatility': float(returns.std() * np.sqrt(252)),
                'num_trades': len(trades),
                'win_rate': self._calculate_win_rate(trades)
            }
            
            return {
                'status': 'success',
                'performance': performance,
                'equity_curve': equity_df.to_dict(orient='records'),
                'trades': trades,
                'positions': positions
            }
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            traceback.print_exc()
            return {'status': 'error', 'reason': str(e)}

    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve"""
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        return abs(drawdown.min())

    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades list"""
        if not trades:
            return 0.0
            
        profits = [
            t['price'] - self.last_trade_price if t['action'] == 'sell' else 
            self.last_trade_price - t['price'] if t['action'] == 'buy' else 0
            for t in trades if self.last_trade_price is not None
        ]
        
        winning_trades = sum(1 for p in profits if p > 0)
        return winning_trades / len(profits) if profits else 0.0

    def optimize_hyperparameters(self, price_data: pd.DataFrame, param_space: Dict = None) -> Dict:
        """Run hyperparameter optimization using Ray Tune"""
        try:
            # Default parameter space if not provided
            if param_space is None:
                param_space = {
                    "lookback_window": tune.choice([50, 100, 150, 200]),
                    "prediction_horizon": tune.choice([1, 3, 5, 10]),
                    "risk_limit": tune.uniform(0.01, 0.05),
                    "max_position": tune.uniform(0.5, 2.0),
                    "transaction_cost": tune.uniform(0.0001, 0.001)
                }
                
            # Define evaluation function for Ray Tune
            def evaluate_params(config):
                # Create strategy instance with config
                strategy = QuantumEnhancedStrategy(config)
                
                # Run backtest
                result = strategy.backtest(price_data)
                
                if result['status'] == 'error':
                    # Return very negative score for failed configs
                    tune.report(score=-1000)
                    return
                    
                # Calculate score (could be Sharpe ratio or other metric)
                score = result['performance']['sharpe_ratio']
                
                # Report score to Ray Tune
                tune.report(
                    score=score,
                    sharpe=result['performance']['sharpe_ratio'],
                    returns=result['performance']['total_return'],
                    drawdown=result['performance']['max_drawdown'],
                    trades=result['performance']['num_trades']
                )
                
            # Configure search algorithm and scheduler
            search_alg = HyperOptSearch()
            scheduler = ASHAScheduler(
                max_t=100,
                grace_period=10,
                reduction_factor=2
            )
            
            # Run optimization
            analysis = tune.run(
                evaluate_params,
                config=param_space,
                num_samples=20,
                search_alg=search_alg,
                scheduler=scheduler,
                resources_per_trial={"cpu": 2},
                verbose=1
            )
            
            # Get best configuration
            best_config = analysis.get_best_config(metric="score", mode="max")
            best_result = analysis.best_result
            
            return {
                "status": "success",
                "best_config": best_config,
                "best_score": best_result["score"],
                "all_results": analysis.results
            }
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return {'status': 'error', 'reason': str(e)}

    def run_live_trading(self, data_source, broker_api, run_duration: int = 3600):
        """Run live trading for specified duration in seconds"""
        try:
            logger.info(f"Starting live trading for {run_duration} seconds")
            
            start_time = time.time()
            last_trade_time = 0
            minimum_trade_interval = 300  # 5 minutes between trades
            
            # Trading loop
            while time.time() - start_time < run_duration:
                try:
                    # Fetch latest market data
                    market_data = data_source.get_latest_data()
                    
                    # Skip if not enough data
                    if len(market_data) < self.lookback_window + 10:
                        logger.warning("Insufficient market data, waiting for more data")
                        time.sleep(60)
                        continue
                    
                    # Get current position from broker
                    current_position = broker_api.get_position()
                    
                    # Update self.position with actual position
                    self.position = current_position
                    
                    # Generate predictions
                    predictions = self.predict(market_data)
                    
                    if 'error' in predictions:
                        logger.warning(f"Prediction error: {predictions['error']}")
                        time.sleep(60)
                        continue
                    
                    # Check if we should generate a signal
                    current_time = time.time()
                    if current_time - last_trade_time < minimum_trade_interval:
                        time.sleep(10)
                        continue
                    
                    # Generate trading signal
                    signal = self.generate_trade_signal(predictions, current_position)
                    
                    # Execute trade if not hold
                    if signal['action'] != 'hold':
                        # Execute trade via broker API
                        order_result = broker_api.place_order(
                            action=signal['action'],
                            quantity=signal['quantity'],
                            order_type='market'
                        )
                        
                        logger.info(f"Order placed: {order_result}")
                        last_trade_time = current_time
                    
                    # Sleep before next iteration
                    time.sleep(10)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {str(e)}")
                    time.sleep(60)
            
            logger.info("Live trading session completed")
            return {"status": "completed", "run_duration": run_duration}
            
        except Exception as e:
            logger.error(f"Fatal error in live trading: {str(e)}")
            return {'status': 'error', 'reason': str(e)}

    def save_model(self, path: str) -> bool:
        """Save the model and strategy state"""
        try:
            # Create directory if not exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model weights separately
            if self.model:
                self.model.save_weights(f"{path}_weights")
            
            # Prepare strategy state (excluding model)
            model_backup = self.model
            self.model = None
            
            # Save strategy object
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            
            # Restore model
            self.model = model_backup
            
            logger.info(f"Strategy saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    @classmethod
    def load_model(cls, path: str):
        """Load saved model and strategy state"""
        try:
            # Load strategy object
            with open(path, 'rb') as f:
                strategy = pickle.load(f)
            
            # Load model weights if they exist
            weights_path = f"{path}_weights"
            if os.path.exists(weights_path) and strategy.trained:
                # Recreate model architecture
                input_dim = strategy.feature_extractor.n_features
                strategy._create_model(input_dim)
                
                # Load weights
                strategy.model.load_weights(weights_path)
            
            logger.info(f"Strategy loaded from {path}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

    def __str__(self) -> str:
        """Return string representation of the strategy"""
        status = "Trained" if self.trained else "Not trained"
        return f"QuantumEnhancedStrategy(status={status}, position={self.position:.2f})"
    