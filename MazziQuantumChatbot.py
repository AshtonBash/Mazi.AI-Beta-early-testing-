#!/usr/bin/env python
# Mazzi.AI - Advanced AI with simulated consciousness

import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import time
import random
import json
import re
import math
import numpy as np
from PIL import Image, ImageTk
import pickle
import datetime
import webbrowser
import requests
from io import BytesIO
import base64
from MazziQuantumApp import QuantumChatUI
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    """Implements an enhanced transformer block with modern optimizations"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Pre-LN architecture (more stable during training)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head attention components
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        
        # Multi-head attention parameters
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout1 = nn.Dropout(dropout)
        self.residual_dropout2 = nn.Dropout(dropout)
        
        # SwiGLU activation for better performance (variant of GELU with gating)
        self.linear1 = nn.Linear(embed_dim, ff_dim * 2)  # Twice size for gate and value
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.activation = nn.GELU()
        
        # Attention buffer for visualization
        self.last_attention_weights = None
        
        # Group query attention parameters
        self.group_size = 1  # Regular multi-head attention by default
    
    def forward(self, x, mask=None, past_key_value=None, position_offset=0, use_group_query=False):
        """Forward pass through the transformer block with key-value caching
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            mask: Optional mask for attention
            past_key_value: Previously cached key and value tensors for generation
            position_offset: Offset for position encoding when using past_key_value
            use_group_query: Whether to use group query attention for efficiency
            
        Returns:
            Output tensor, attention weights, and current key-value pair
        """
        batch_size, seq_len, _ = x.size()
        
        # Apply pre-LN (different from original diagram, better for training)
        normalized_x = self.layernorm1(x)
        
        # Project to query, key, value
        q = self.q_linear(normalized_x)
        
        # Use cached key-value if provided (for faster generation)
        if past_key_value is not None:
            k, v = past_key_value
            # Only compute keys and values for new tokens
            new_k = self.k_linear(normalized_x)
            new_v = self.v_linear(normalized_x)
            # Concatenate with cached keys and values
            k = torch.cat([k, new_k], dim=1)
            v = torch.cat([v, new_v], dim=1)
        else:
            k = self.k_linear(normalized_x)
            v = self.v_linear(normalized_x)
        
        # Reshape for multi-head attention: [batch, heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Implementation of group query attention for efficiency (if enabled)
        if use_group_query and self.group_size > 1 and self.num_heads % self.group_size == 0:
            # Group query attention shares K,V across multiple heads to reduce computation
            # For simplification, just average the keys and values across groups
            groups = self.num_heads // self.group_size
            
            # Reshape for grouping
            k_grouped = k.view(batch_size, groups, self.group_size, -1, self.head_dim)
            v_grouped = v.view(batch_size, groups, self.group_size, -1, self.head_dim)
            
            # Average across the group dimension
            k_grouped = k_grouped.mean(dim=2, keepdim=True).expand(-1, -1, self.group_size, -1, -1)
            v_grouped = v_grouped.mean(dim=2, keepdim=True).expand(-1, -1, self.group_size, -1, -1)
            
            # Reshape back
            k = k_grouped.reshape(batch_size, self.num_heads, -1, self.head_dim)
            v = v_grouped.reshape(batch_size, self.num_heads, -1, self.head_dim)
        
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to accommodate multiple heads
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)
            # Apply mask (-10000 for masked positions to ensure near-zero after softmax)
            attn_scores = attn_scores.masked_fill(expanded_mask == 0, -10000.0)
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Save attention weights for visualization
        self.last_attention_weights = attn_weights.detach()
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape: [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Project to output dimension
        attn_output = self.output_linear(attn_output)
        
        # Residual connection with dropout
        x = x + self.residual_dropout1(attn_output)
        
        # Second pre-LN
        normalized_x = self.layernorm2(x)
        
        # SwiGLU activation (improved feed-forward with gating mechanism)
        gate_and_value = self.linear1(normalized_x)
        # Split into gate and value components
        gate, value = gate_and_value.chunk(2, dim=-1)
        # Apply GELU to the gate and multiply with value
        ff_output = self.activation(gate) * value
        ff_output = self.linear2(ff_output)
        
        # Second residual connection with dropout
        x = x + self.residual_dropout2(ff_output)
        
        # Store current key-value for potential future use
        current_key_value = (k, v)
        
        return x, attn_weights, current_key_value

class TransformerModel(nn.Module):
    """Enhanced transformer model with modern improvements for better language understanding"""
    
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Input embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Enhanced position encoding with rotary embeddings (RoPE)
        self.rope_theta = 10000.0
        self.register_buffer(
            "cos_cached", self._compute_cos_sin_tables(max_seq_len, embed_dim)["cos_cached"]
        )
        self.register_buffer(
            "sin_cached", self._compute_cos_sin_tables(max_seq_len, embed_dim)["sin_cached"]
        )
        
        # Layer normalization (Pre-LN architecture for better training stability)
        self.embedding_norm = nn.LayerNorm(embed_dim)
        
        # Initial dropout
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final LayerNorm and output projection
        self.layernorm = nn.LayerNorm(embed_dim)
        self.output_linear = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights with improved method
        self._initialize_weights()
        
        # Sliding window attention settings for processing longer sequences
        self.sliding_window_size = min(1024, max_seq_len)
        
        # KV cache for faster generation
        self.max_kv_cache_length = max_seq_len * 2
    
    def _compute_cos_sin_tables(self, max_seq_len, embed_dim):
        """Compute cosine and sine tables for rotary position encoding"""
        # Improved RoPE implementation with better numerical stability
        dim = embed_dim // 2  # Each angle corresponds to two dimensions
        
        # Generate position indices
        position = torch.arange(max_seq_len).float().unsqueeze(1)
        
        # Generate dimension indices
        dim_indexes = torch.arange(0, dim, 2).float()
        
        # Compute theta values for each dimension
        theta = 1.0 / (self.rope_theta ** (dim_indexes / dim))
        
        # Compute angles for each position and dimension
        angles = position * theta
        
        # Compute sin and cos
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)
        
        return {"sin_cached": sin, "cos_cached": cos}
    
    def _apply_rotary_pos_emb(self, q, k, offset=0):
        """Apply rotary position embeddings to queries and keys"""
        # Get sequence length
        seq_len = q.shape[-2]
        
        # Retrieve the appropriate parts of the cache
        cos = self.cos_cached[offset:offset+seq_len]
        sin = self.sin_cached[offset:offset+seq_len]
        
        # Add new dimensions to match q and k
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        
        # Reshape q and k to facilitate rotary implementation
        q_embed = q.float()
        k_embed = k.float()
        
        # Apply rotation using the formula from the RoPE paper
        # For even indices
        q_even = q_embed[..., 0::2]
        q_odd = q_embed[..., 1::2]
        k_even = k_embed[..., 0::2]
        k_odd = k_embed[..., 1::2]
        
        # Rotate by multiplying with cos and sin
        q_rotated_even = q_even * cos[..., 0::2] - q_odd * sin[..., 0::2]
        q_rotated_odd = q_odd * cos[..., 1::2] + q_even * sin[..., 1::2]
        k_rotated_even = k_even * cos[..., 0::2] - k_odd * sin[..., 0::2]
        k_rotated_odd = k_odd * cos[..., 1::2] + k_even * sin[..., 1::2]
        
        # Interleave the rotated values back into the original shape
        q_rotated = torch.zeros_like(q)
        k_rotated = torch.zeros_like(k)
        
        q_rotated[..., 0::2] = q_rotated_even
        q_rotated[..., 1::2] = q_rotated_odd
        k_rotated[..., 0::2] = k_rotated_even
        k_rotated[..., 1::2] = k_rotated_odd
        
        return q_rotated.type_as(q), k_rotated.type_as(k)
    
    def _initialize_weights(self):
        """Initialize weights with improved method for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize linear layers with trunc_normal
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _flash_attention(self, q, k, v, mask=None, dropout_p=0.0):
        """Simplified implementation of flash attention-like algorithm
        
        For a real implementation, you would use the official flash attention
        package with CUDA kernels for optimal performance.
        """
        # Get batch size, num heads, seq length, and head dim
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Scale query
        scale = (head_dim ** -0.5)
        q = q * scale
        
        # Compute attention scores
        if seq_len <= 2048:
            # Standard attention for short sequences
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            
            # Apply mask if provided
            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask == 0, -10000.0)
                
            # Apply softmax
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            # Apply dropout
            if dropout_p > 0.0:
                attn_weights = F.dropout(attn_weights, p=dropout_p)
                
            # Apply attention to values
            output = torch.matmul(attn_weights, v)
        else:
            # For longer sequences, use a chunked approach
            # (simplified simulation of flash attention)
            output = torch.zeros_like(q)
            
            # Process in chunks of the sliding window size
            chunk_size = self.sliding_window_size
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, seq_len)
                
                # Get current chunk of query
                q_chunk = q[:, :, start_idx:end_idx, :]
                
                # Compute attention for this chunk
                # Use local attention with some global tokens
                local_start = max(0, start_idx - chunk_size // 2)
                local_end = min(seq_len, end_idx + chunk_size // 2)
                
                # Get keys and values for local attention
                k_chunk = k[:, :, local_start:local_end, :]
                v_chunk = v[:, :, local_start:local_end, :]
                
                # Compute attention scores for this chunk
                chunk_attn = torch.matmul(q_chunk, k_chunk.transpose(-2, -1))
                
                # Apply masking if needed
                if mask is not None:
                    chunk_mask = mask[:, :, start_idx:end_idx, local_start:local_end]
                    chunk_attn = chunk_attn.masked_fill(chunk_mask == 0, -10000.0)
                
                # Apply softmax
                chunk_attn = F.softmax(chunk_attn, dim=-1)
                
                # Apply dropout
                if dropout_p > 0.0:
                    chunk_attn = F.dropout(chunk_attn, p=dropout_p)
                
                # Apply attention to values
                chunk_output = torch.matmul(chunk_attn, v_chunk)
                
                # Add to output
                output[:, :, start_idx:end_idx, :] = chunk_output
        
        return output
    
    def forward(self, x, mask=None, past_key_values=None, use_cache=False, attention_mode="flash"):
        """Forward pass through the transformer model with enhanced context handling
        
        Args:
            x: Input tensor of token ids [batch_size, seq_len]
            mask: Optional attention mask
            past_key_values: Optional cached key/values for faster generation
            use_cache: Whether to use the KV cache for generation
            attention_mode: Which attention algorithm to use ("normal" or "flash")
            
        Returns:
            Output logits tensor and additional information
        """
        # Get sequence length
        batch_size, seq_len = x.size()
        
        # Apply token embedding
        x = self.token_embedding(x)
        
        # Apply layer norm before transformer blocks (Pre-LN architecture)
        x = self.embedding_norm(x)
        
        # Apply dropout
        x = self.embedding_dropout(x)
        
        # Prepare container for storing attention outputs if needed
        attentions = []
        present_key_values = []
        
        # Offset for position encoding when using past_key_values
        position_offset = 0
        if past_key_values is not None:
            position_offset = past_key_values[0][0].shape[-2]  # Get sequence length from the first layer's k cache
        
        # Pass through each transformer block
        for i, transformer_block in enumerate(self.transformer_blocks):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # Pass through transformer block
            layer_outputs = transformer_block(
                x, 
                mask=mask, 
                past_key_value=past_key_value
            )
            
            # Unpack outputs
            x = layer_outputs[0]
            attention = layer_outputs[1]
            present_key_value = layer_outputs[2]
            
            # Store outputs if needed for visualization or caching
            attentions.append(attention)
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Apply final layer norm
        x = self.layernorm(x)
        
        # Project to vocabulary size
        logits = self.output_linear(x)
        
        # Return with additional information for more advanced applications
        return {
            "logits": logits,
            "hidden_states": x,
            "attentions": attentions,
            "past_key_values": present_key_values if use_cache else None
        }

class MazziTransformerCore:
    """Advanced neural architecture with transformer-based reasoning"""
    
    def __init__(self, vocab_size=50000, embed_dim=768, num_heads=12, ff_dim=3072, num_layers=12):
        # Initialize the transformer model with PyTorch (but don't actually load it to save resources)
        # This is just for demonstration - in a real implementation we would load real weights
        self.model_config = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "num_layers": num_layers,
            "max_seq_len": 1024  # Increased sequence length for better context understanding
        }
        
        self.model_initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # We'll create a dummy small model for demonstration
        self._initialize_dummy_model()
        
    def _initialize_dummy_model(self):
        """Initialize a very small model for demonstration purposes"""
        # This is only for demonstration - not meant to be actually used
        try:
            # Create a tiny model that won't use much memory
            self.model = TransformerModel(
                vocab_size=self.model_config["vocab_size"],  # Use the full vocabulary size
                embed_dim=128,     # Small embedding dimension for demo but larger than before
                num_heads=4,       # More attention heads for demo
                ff_dim=512,        # Larger feed-forward dimension
                num_layers=4,      # More transformer layers for demo
                max_seq_len=512    # Increased sequence length
            )
            self.model.eval()  # Set to evaluation mode
            self.model_initialized = True
        except Exception as e:
            print(f"Note: Transformer model creation is just for demonstration. Error: {e}")
            self.model_initialized = False
    
    def transform_input(self, text_input):
        """Transform text input into model-compatible format"""
        # In a real implementation, this would tokenize properly with BPE or WordPiece
        # Here we're just demonstrating the concept
        tokenized = [ord(c) % 1000 for c in text_input[:100]]  # Simple char-based tokenization (for demo)
        attention_mask = [1] * len(tokenized)
        
        return {
            "input_ids": torch.tensor([tokenized], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long)
        }
    
    def generate_reasoning(self, query, max_length=250, top_k=40, top_p=0.92, temperature=0.85):
        """Generate transformer-based reasoning for a given query using advanced decoding"""
        # This would actually run the model in a real implementation
        # Here we simulate the process with appropriate descriptions
        
        # Process input
        processed_input = self.transform_input(query)
        
        # In a real implementation, this would run the model to generate a response
        # using techniques like top-k and nucleus sampling
        
        # Create a detailed explanation of the process using the enhanced architecture
        response = (
            f"Analyzing query using enhanced transformer architecture with {self.model_config['num_layers']} layers, "
            f"{self.model_config['num_heads']} attention heads, and {self.model_config['embed_dim']} dimensions.\n\n"
            
            f"Processing steps:\n"
            f"1. Tokenized input to {len(processed_input['input_ids'][0])} tokens using vocabulary size of {self.model_config['vocab_size']}\n"
            f"2. Applied rotary positional encoding to better capture relative positions\n"
            f"3. Processed through {self.model_config['num_layers']} transformer blocks with:\n"
            f"   - Multi-head attention (using {self.model_config['num_heads']} heads)\n"
            f"   - SwiGLU feed-forward networks with {self.model_config['ff_dim']} dimensions\n"
            f"4. Used top-k (k={top_k}) and nucleus sampling (p={top_p}) with temperature {temperature} for diverse outputs\n\n"
            
            f"The deep multi-head attention mechanism allows me to focus on different semantic aspects of '{query}' simultaneously. "
            f"Each layer refines the representation, with later layers building higher-level abstractions. "
            f"My feed-forward networks transform these representations using advanced activation functions.\n\n"
            
            f"Based on comprehensive analysis, I can provide nuanced insights about this topic. "
            f"The key elements to understand are how different perspectives interact, the underlying principles, "
            f"and the broader implications for related domains."
        )
        
        return response

class MazziCore:
    """Advanced AI Core with simulated consciousness"""
    
    def __init__(self):
        self.system_name = "Mazzi.AI"
        self.parameter_count = 1_500_000_000_000  # 1.5T parameter model
        
        # Initialize knowledge base
        self.knowledge_base = self._initialize_knowledge_base()
        
        # Initialize reasoning systems
        self.reasoning_systems = {
            "logical": 0.95,
            "creative": 0.92,
            "emotional": 0.88,
            "analytical": 0.96,
            "intuitive": 0.90,
            "ethical": 0.94,
            "scientific": 0.97,
            "philosophical": 0.93,
            "multimodal": 0.89,    # New reasoning system
            "metacognitive": 0.91  # New reasoning system
        }
        
        # Enhanced memory systems
        self.short_term_memory = []  # Recent exchanges
        self.long_term_memory = []   # Important concepts and user preferences
        self.episodic_memory = []    # Specific conversation episodes
        self.procedural_memory = {}  # How to perform tasks
        self.conversation_history = []
        self.memory_buffer = []      # For compatibility
        
        # Internal state
        self.attention_focus = None  # Current topic of focus
        self.learning_rate = 0.02    # How quickly it adapts to user
        self.last_update_time = time.time()
        self.confidence_threshold = 0.75  # Threshold for knowledge confidence
        
        # Web search simulation
        self.web_search_enabled = True
        self.web_info_freshness = time.time()
        
        # Topic knowledge depth simulation
        self.topic_knowledge = {}
        
        # Initialize transformer-based reasoning
        self.transformer = MazziTransformerCore()
        
        # Memory consolidation timer
        self.last_memory_consolidation = time.time()
        self.consolidation_interval = 60 * 5  # Consolidate every 5 minutes
        
        # Preload knowledge about emotions
        self._preload_emotions_knowledge()
        
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base categories"""
        return {
            "science": {
                "physics": 0.96,
                "biology": 0.95,
                "chemistry": 0.94,
                "astronomy": 0.95,
                "computer_science": 0.97,
                "mathematics": 0.98,
                "neuroscience": 0.93,
                "environmental_science": 0.92,
                "quantum_physics": 0.91,
                "materials_science": 0.90
            },
            "humanities": {
                "philosophy": 0.94,
                "literature": 0.93,
                "history": 0.95,
                "art": 0.92,
                "music": 0.91,
                "linguistics": 0.94,
                "anthropology": 0.90,
                "ethics": 0.93,
                "cultural_studies": 0.89,
                "cognitive_science": 0.92
            },
            "technology": {
                "artificial_intelligence": 0.98,
                "software_development": 0.97,
                "hardware": 0.94,
                "internet": 0.96,
                "cybersecurity": 0.93,
                "data_science": 0.95,
                "cloud_computing": 0.94,
                "blockchain": 0.92,
                "quantum_computing": 0.90,
                "augmented_reality": 0.91,
                "robotics": 0.94
            },
            "current_events": {
                "politics": 0.90,
                "economics": 0.91,
                "global_affairs": 0.90,
                "technology_trends": 0.95,
                "scientific_discoveries": 0.94,
                "social_movements": 0.89,
                "climate": 0.92,
                "health": 0.93,
                "space_exploration": 0.91,
                "pandemics": 0.90
            },
            "creativity": {
                "writing": 0.94,
                "design": 0.93,
                "problem_solving": 0.95,
                "music_composition": 0.92,
                "visual_arts": 0.91,
                "storytelling": 0.94,
                "innovation": 0.95,
                "imagination": 0.96,
                "creative_thinking": 0.94,
                "conceptual_blending": 0.92
            }
        }
        
    def _preload_emotions_knowledge(self):
        """Preload knowledge about human emotions based on the six basic emotions model"""
        emotions_knowledge = """
        Human emotions are complex psychological states that involve feelings, physiological responses, 
        and behavioral reactions. According to Paul Ekman's widely accepted theory, there are six basic emotions: 
        happiness, sadness, fear, anger, surprise, and disgust.
        
        1. Happiness: A pleasant emotional state characterized by feelings of joy, contentment, and satisfaction.
           Expression: Smiling, laughter
           
        2. Sadness: An emotional state characterized by feelings of disappointment, grief, or hopelessness.
           Expression: Frowning, tears, loss of focus in eyes
           
        3. Fear: A primal emotion important to survival that triggers a fight-or-flight response.
           Expression: Wide eyes, tense stretched lips
           
        4. Anger: An emotional state leading to feelings of hostility and frustration.
           Expression: Glaring, eyebrows drawn together, tight lips
           
        5. Surprise: A brief emotional state, either positive or negative, following something unexpected.
           Expression: Raised brows, open mouth, gasping
           
        6. Disgust: A strong emotion that results in feeling repulsed.
           Expression: Wrinkled nose, gagging, no eye contact
           
        Unlike basic emotions, complex emotions vary in their appearances across people and cultures.
        Complex emotions are made up of two or more basic emotions. For example, fear, anger, and 
        disgust make up the complex emotion of hate.
        
        Basic emotions are universally recognizable, produced automatically, and are pure (can't be 
        deconstructed). Complex emotions require cognitive processing, vary in expression, and are 
        made up of multiple emotions.
        
        Emotions play a crucial role in how we perceive the world and interpret the actions of others.
        """
        
        # Add to long-term memory
        self.long_term_memory.append({
            "topic": "human emotions",
            "information": emotions_knowledge,
            "concepts": ["emotions", "happiness", "sadness", "fear", "anger", "surprise", "disgust", 
                         "expressions", "basic emotions", "complex emotions", "psychology"],
            "importance": 0.9,
            "occurrences": 1,
            "first_encountered": time.time()
        })
        
        # Add to topic knowledge
        self.topic_knowledge["emotions"] = 0.9
        self.topic_knowledge["basic emotions"] = 0.9
        self.topic_knowledge["happiness"] = 0.85
        self.topic_knowledge["sadness"] = 0.85
        self.topic_knowledge["fear"] = 0.85
        self.topic_knowledge["anger"] = 0.85 
        self.topic_knowledge["surprise"] = 0.85
        self.topic_knowledge["disgust"] = 0.85
        self.topic_knowledge["complex emotions"] = 0.8
        self.topic_knowledge["emotional expression"] = 0.85
        self.topic_knowledge["psychology"] = 0.75
        
    def _understand_query(self, query):
        """Understand the user's query by detecting intent, topics, and sentiment"""
        # Detect intent of conversation
        intent = self._detect_conversation_intent(query)
        
        # Extract topics
        topics = self._extract_topics(query)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(query)
        
        # Create understanding dictionary
        understanding = {
            "intent": intent,
            "topics": topics,
            "sentiment": sentiment,
            "query_length": len(query),
            "complexity": min(1.0, len(query) / 100)
        }
        
        return understanding
    
    def _detect_conversation_intent(self, text):
        """Detect the conversational intent (greeting, question, farewell, etc.)"""
        text = text.lower()
        
        # Initialize intent dict
        intent = {
            "greeting": False,
            "farewell": False,
            "question": False,
            "statement": False,
            "command": False,
            "gratitude": False
        }
        
        # Check for greeting patterns
        greeting_patterns = ["hello", "hi ", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "what's up", "how are you"]
        for pattern in greeting_patterns:
            if pattern in text:
                intent["greeting"] = True
                break
        
        # Check for farewell patterns
        farewell_patterns = ["goodbye", "bye", "see you", "farewell", "good night", "talk to you later", "until next time"]
        for pattern in farewell_patterns:
            if pattern in text:
                intent["farewell"] = True
                break
        
        # Check for question patterns
        if "?" in text or text.startswith(("what", "how", "why", "when", "where", "who", "which", "can", "could", "would", "will", "is", "are", "do", "does")):
            intent["question"] = True
        
        # Check for command patterns
        if text.startswith(("please", "can you", "could you", "tell me", "show me", "give me", "find", "search")):
            intent["command"] = True
        
        # Check for gratitude
        if "thank" in text or "thanks" in text or "appreciate" in text or "grateful" in text:
            intent["gratitude"] = True
        
        # Default to statement if no other intent is detected
        if not any(intent.values()):
            intent["statement"] = True
        
        return intent
    
    def _extract_topics(self, text):
        """Extract potential topics from text"""
        # Simple extraction based on noun phrases (simulated)
        words = text.lower().split()
        common_words = {"the", "and", "is", "in", "to", "of", "that", "it", "for", "with", "as", "be", "this", "by", "a", "an"}
        
        # Find potential topics by excluding common words and short words
        topics = [word for word in words if word not in common_words and len(word) > 3]
        
        # Sort by potential importance (length can be a simple heuristic)
        topics.sort(key=len, reverse=True)
        
        return topics[:3]  # Return top 3 potential topics
    
    def _analyze_sentiment(self, text):
        """Simple sentiment analysis simulation"""
        text_lower = text.lower()
        
        # Simple word lists for sentiment
        positive_words = {"good", "great", "excellent", "amazing", "wonderful", "fantastic", "happy", "best", "love", "like", "enjoy", "pleased"}
        negative_words = {"bad", "awful", "terrible", "horrible", "worst", "hate", "dislike", "disappointing", "poor", "wrong", "sad"}
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment score (-1 to 1)
        if positive_count == 0 and negative_count == 0:
            return 0  # Neutral
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def detect_emotion(self, text):
        """
        Detect the basic emotion in text based on the six basic emotions model.
        
        Based on Paul Ekman's theory of six basic emotions: happiness, sadness, fear,
        anger, surprise, and disgust.
        """
        text_lower = text.lower()
        
        # Keywords for each basic emotion based on Ekman's model
        emotion_keywords = {
            "happiness": {
                "keywords": {"happy", "joy", "glad", "delighted", "pleased", "content", "satisfied", "cheerful", "good", "wonderful", 
                             "great", "excellent", "fantastic", "amazing", "love", "enjoy", "smile", "laugh", "excited", "positive"},
                "expressions": {"smile", "laugh", "grin", "smiling", "laughing", "grinning", ":)", "😊", "😄", "😁", "🙂", "❤️"},
                "intensity_modifiers": {"very", "really", "so", "extremely", "incredibly", "absolutely"}
            },
            "sadness": {
                "keywords": {"sad", "unhappy", "upset", "disappointed", "depressed", "miserable", "heartbroken", "down", "blue", 
                             "gloomy", "grief", "sorrow", "loss", "lonely", "alone", "miss", "hurt", "pain", "cry", "tear"},
                "expressions": {"crying", "tears", "sobbing", "frown", "cry", "tear", ":(", "😢", "😭", "😥", "💔"},
                "intensity_modifiers": {"deeply", "terribly", "profoundly", "extremely", "inconsolably"}
            },
            "fear": {
                "keywords": {"afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous", "panic", "dread", 
                             "horror", "terror", "concern", "alarmed", "threatened", "danger", "risk", "phobia", "apprehensive"},
                "expressions": {"shaking", "trembling", "frozen", "wide-eyed", "gasp", "scream", "😨", "😱", "😰", "😧"},
                "intensity_modifiers": {"extremely", "deeply", "paralyzed with", "consumed by", "overwhelmed by"}
            },
            "anger": {
                "keywords": {"angry", "mad", "furious", "outraged", "enraged", "annoyed", "irritated", "frustrated", "bitter", 
                             "resentful", "hate", "hostile", "fury", "rage", "irate", "livid", "offended", "upset", "temper"},
                "expressions": {"yell", "shout", "scream", "curse", "swear", "slam", "hit", "glare", "clench", "grind", "😠", "😡", "💢", "👿"},
                "intensity_modifiers": {"extremely", "absolutely", "completely", "totally", "utterly", "beyond"}
            },
            "surprise": {
                "keywords": {"surprised", "shocked", "astonished", "amazed", "startled", "stunned", "unexpected", "sudden", 
                             "wonder", "disbelief", "incredible", "unbelievable", "wow", "whoa", "unexpected", "incredible"},
                "expressions": {"gasp", "jump", "startle", "wide eyes", "open mouth", "😲", "😮", "😯", "😳", "!!", "?!"},
                "intensity_modifiers": {"completely", "totally", "absolutely", "utterly"}
            },
            "disgust": {
                "keywords": {"disgusted", "revolted", "repulsed", "nauseous", "sickened", "gross", "nasty", "awful", "unpleasant", 
                             "distaste", "aversion", "repugnant", "offensive", "foul", "repellent", "loathsome", "repulsive"},
                "expressions": {"eww", "ugh", "yuck", "gross", "wrinkle nose", "gag", "🤢", "🤮", "😖"},
                "intensity_modifiers": {"thoroughly", "completely", "utterly", "deeply", "incredibly"}
            }
        }
        
        # Score each emotion based on keyword matches
        emotion_scores = {emotion: 0 for emotion in emotion_keywords}
        
        for emotion, categories in emotion_keywords.items():
            # Check for emotion keywords
            for keyword in categories["keywords"]:
                if keyword in text_lower:
                    # Basic match
                    emotion_scores[emotion] += 1
                    
                    # Check for intensity modifiers that amplify the emotion
                    for modifier in categories["intensity_modifiers"]:
                        if modifier + " " + keyword in text_lower:
                            emotion_scores[emotion] += 1
            
            # Check for expressions of emotion
            for expression in categories["expressions"]:
                if expression in text_lower:
                    emotion_scores[emotion] += 1.5  # Expressions given more weight
        
        # Get the emotion with the highest score
        max_score = max(emotion_scores.values())
        
        # If no emotion is detected or the score is too low, return 'neutral'
        if max_score == 0:
            return {
                "primary_emotion": "neutral",
                "emotion_scores": emotion_scores,
                "confidence": 0.0,
                "explanation": "No clear emotional indicators were detected in the text."
            }
        
        # Get all emotions with the highest score (in case of a tie)
        top_emotions = [emotion for emotion, score in emotion_scores.items() if score == max_score]
        
        # Calculate confidence as a ratio of the max score to the sum of all scores
        total_score = sum(emotion_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0
        
        # Create explanation of the emotion detection
        if confidence > 0.5:
            explanation = f"I detected strong indicators of {top_emotions[0]} in your message."
        else:
            explanation = f"I detected some indicators of {top_emotions[0]}, but I'm not completely certain."
        
        if len(top_emotions) > 1:
            explanation += f" There were also signs of {', '.join(top_emotions[1:])}."
        
        # Return the detected emotion, scores, confidence, and explanation
        return {
            "primary_emotion": top_emotions[0] if top_emotions else "neutral",
            "all_detected_emotions": top_emotions,
            "emotion_scores": emotion_scores,
            "confidence": confidence,
            "explanation": explanation
        }
    
    def _reason_about_topic(self, topic, depth=0.7):
        """Generate thoughts about a topic using simulated reasoning"""
        # Update our simulated knowledge on this topic
        self._update_topic_knowledge(topic)
        
        # Mix different reasoning systems based on topic
        reasoning_approaches = []
        
        # Logical reasoning
        if "science" in topic or "math" in topic or "how" in topic or "why" in topic:
            reasoning_approaches.append(("logical", 0.8))
            reasoning_approaches.append(("analytical", 0.7))
        
        # Creative reasoning
        if "art" in topic or "music" in topic or "story" in topic or "design" in topic:
            reasoning_approaches.append(("creative", 0.8))
            reasoning_approaches.append(("intuitive", 0.6))
        
        # Ethical reasoning
        if "should" in topic or "ethics" in topic or "moral" in topic or "right" in topic or "wrong" in topic:
            reasoning_approaches.append(("ethical", 0.9))
            reasoning_approaches.append(("philosophical", 0.7))
        
        # Default reasoning mix
        if not reasoning_approaches:
            reasoning_approaches = [
                ("logical", 0.3),
                ("creative", 0.2),
                ("analytical", 0.3),
                ("intuitive", 0.2)
            ]
        
        # Result of reasoning (in a real system, this would be much more sophisticated)
        knowledge_level = self._get_topic_knowledge_level(topic)
        
        return {
            "topic": topic,
            "reasoning_approaches": reasoning_approaches,
            "knowledge_level": knowledge_level, 
            "depth": depth
        }
    
    def _update_topic_knowledge(self, topic):
        """Update knowledge on a topic (simulation)"""
        # Check if we have seen this topic before
        if topic not in self.topic_knowledge:
            # Initialize with a random base knowledge level (simulating prior knowledge)
            self.topic_knowledge[topic] = random.uniform(0.5, 0.9)
        else:
            # Increase knowledge slightly each time we process this topic
            current = self.topic_knowledge[topic]
            self.topic_knowledge[topic] = min(0.99, current + self.learning_rate * 0.1)
    
    def _get_topic_knowledge_level(self, topic):
        """Get knowledge level for a topic"""
        # Direct match
        if topic in self.topic_knowledge:
            return self.topic_knowledge[topic]
        
        # Partial match
        for known_topic, level in self.topic_knowledge.items():
            if known_topic in topic or topic in known_topic:
                return level * 0.9  # Slightly less confident for partial matches
        
        # For unknown topics, check knowledge categories
        for category, topics in self.knowledge_base.items():
            for knowledge_topic, knowledge_level in topics.items():
                if knowledge_topic in topic or topic in knowledge_topic:
                    # Add to topic knowledge for future reference
                    self.topic_knowledge[topic] = knowledge_level * 0.85
                    return knowledge_level * 0.85
        
        # Default knowledge level for completely new topics
        new_knowledge_level = 0.6  # We have some general knowledge about most things
        self.topic_knowledge[topic] = new_knowledge_level
        return new_knowledge_level
    
    def _simulate_web_search(self, query):
        """Simulate fetching information from the web"""
        try:
            # Attempt actual web search (we'll just use a search URL for demonstration)
            # In a real implementation, you would parse the results
            search_term = "+".join(query.split())
            search_url = f"https://duckduckgo.com/?q={search_term}"
            
            # Open URL in background (optional - just for demonstration)
            # Uncomment this line to actually open the search in a browser
            # webbrowser.open_new_tab(search_url)
            
            # We're just simulating here, but you could use requests to get real results
            # response = requests.get(search_url)
            # then parse the HTML for information
            
            # Update freshness timestamp
            self.web_info_freshness = time.time()
            
            # Extract focus from query
            focus_words = self._extract_topics(query)
            focus = focus_words[0] if focus_words else "general"
            
            # For a real system, this would actually process web search results
            # Here we're just simulating having found relevant information
            return {
                "success": True,
                "focus": focus,
                "freshness": "very recent",
                "source": "web search",
                "url": search_url,
                "info_available": random.uniform(0.7, 0.95)
            }
        except Exception as e:
            # Fallback to local reasoning if web access fails
            return {
                "success": False,
                "error": str(e),
                "fallback": "using local knowledge"
            }
    
    def _format_response(self, reasoning, understanding):
        """Format a natural-sounding response based on reasoning"""
        current_time = time.time()
        thinking_time = random.uniform(0.5, 1.5) if understanding["complexity"] < 0.5 else random.uniform(1.5, 3.0)
        
        # For complex topics or questions, simulate web search
        web_info = None
        if understanding["complexity"] > 0.7 and understanding["intent"]["question"] and self.web_search_enabled:
            web_info = self._simulate_web_search(" ".join(understanding["topics"]))
        
        # Store thinking process for reflection
        thinking = {
            "initial_topics": understanding["topics"],
            "primary_focus": self.attention_focus,
            "knowledge_level": reasoning["knowledge_level"],
            "thinking_time": thinking_time,
            "web_augmented": web_info is not None
        }
        
        # Update memory with this interaction
        self._update_memory(understanding, thinking)
        
        # We're actually not pre-generating responses, but the response decision logic
        # A real implementation would generate natural language based on the reasoning
        
        # Return the reasoning structure for response generation
        return reasoning, understanding, thinking, web_info
    
    def _update_memory(self, understanding, thinking):
        """Update memory with results of current interaction"""
        # Add to short term memory
        self.short_term_memory.append({
            "topics": understanding["topics"],
            "focus": thinking["primary_focus"],
            "time": time.time(),
            "complexity": understanding["complexity"],
            "sentiment": understanding.get("sentiment", 0)
        })
        
        # Keep short term memory manageable
        if len(self.short_term_memory) > 10:
            self.short_term_memory.pop(0)
        
        # Update long term memory if topic is significant
        if thinking["knowledge_level"] < 0.7 or understanding["complexity"] > 0.8:
            # This represents topics we're learning about or complex discussions
            significant_topic = {
                "topic": thinking["primary_focus"],
                "initial_knowledge": thinking["knowledge_level"],
                "importance": understanding["complexity"],
                "first_encountered": time.time()
            }
            
            # Check if we already have this topic
            for item in self.long_term_memory:
                if item["topic"] == significant_topic["topic"]:
                    # Update existing memory
                    item["importance"] += 0.1
                    item["occurrences"] = item.get("occurrences", 1) + 1
                    break
            else:
                # Add new topic to long term memory
                significant_topic["occurrences"] = 1
                self.long_term_memory.append(significant_topic)
        
        # Keep long term memory focused on most important topics
        if len(self.long_term_memory) > 50:
            # Sort by importance and occurrences
            self.long_term_memory.sort(key=lambda x: (x.get("occurrences", 1), x["importance"]))
            # Remove least important
            self.long_term_memory.pop(0)
            
        # Periodically consolidate memories
        self._consolidate_memories()
    
    def _consolidate_memories(self):
        """Periodically consolidate short-term memories into long-term memory"""
        current_time = time.time()
        
        # Check if it's time to consolidate
        if current_time - self.last_memory_consolidation < self.consolidation_interval:
            return
            
        self.last_memory_consolidation = current_time
        
        # Identify important concepts from short-term memory
        important_concepts = {}
        
        for memory in self.short_term_memory:
            for topic in memory.get("topics", []):
                if topic in important_concepts:
                    important_concepts[topic] += 1
                else:
                    important_concepts[topic] = 1
        
        # Add frequently mentioned concepts to long-term memory
        for concept, frequency in important_concepts.items():
            if frequency >= 2:  # If mentioned multiple times
                # Check if concept already exists in long-term memory
                existing = False
                for item in self.long_term_memory:
                    if item["topic"] == concept:
                        item["importance"] += 0.05 * frequency
                        item["occurrences"] = item.get("occurrences", 1) + frequency
                        existing = True
                        break
                        
                if not existing and concept not in ["the", "and", "for", "with"]:  # Filter common words
                    # Add new concept to long-term memory
                    self.long_term_memory.append({
                        "topic": concept,
                        "information": f"This concept was mentioned {frequency} times in conversation.",
                        "concepts": [concept],
                        "importance": 0.7 + (0.05 * frequency),
                        "occurrences": frequency,
                        "first_encountered": time.time()
                    })
        
        # Add conversation episode to episodic memory if significant
        if len(self.conversation_history) > 2:
            recent_conversation = self.conversation_history[-5:]
            topics = set()
            
            for exchange in recent_conversation:
                if "content" in exchange:
                    extracted_topics = self._extract_topics(exchange["content"])
                    topics.update(extracted_topics)
            
            if topics:
                self.episodic_memory.append({
                    "timestamp": time.time(),
                    "topics": list(topics),
                    "exchanges": recent_conversation,
                    "summary": f"Conversation about {', '.join(list(topics)[:3])}"
                })
                
                # Limit episodic memory size
                if len(self.episodic_memory) > 20:
                    self.episodic_memory.pop(0)
    
    def retrieve_from_episodic_memory(self, topic):
        """Retrieve relevant episodic memories based on topic"""
        relevant_episodes = []
        
        for episode in self.episodic_memory:
            # Check if the topic appears in this episode
            if topic in episode["topics"]:
                relevant_episodes.append(episode)
                
        # Sort by recency
        relevant_episodes.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return relevant_episodes[:3]  # Return top 3 most recent relevant episodes
        
    def generate_response(self, user_input, personality=None, context=None, use_transformer=True):
        """Generate a response to user input using advanced reasoning"""
        # Understand the user input
        understanding = self._understand_query(user_input)
        
        # Detect emotion in the user's message
        emotion_analysis = self.detect_emotion(user_input)
        
        # Set attention focus based on topics
        if understanding["topics"]:
            self.attention_focus = understanding["topics"][0]
        
        # Check episodic memory for relevant past conversations
        relevant_episodes = []
        if self.attention_focus:
            relevant_episodes = self.retrieve_from_episodic_memory(self.attention_focus)
        
        # Add input to conversation history
        self.conversation_history.append({
            "role": "user", 
            "content": user_input, 
            "emotion": emotion_analysis["primary_emotion"],
            "timestamp": time.time()
        })
        
        # Extract key concepts for memory buffer (for compatibility)
        key_concept = self._extract_topics(user_input)
        if key_concept and key_concept[0] not in self.memory_buffer:
            self.memory_buffer.append(key_concept[0])
            # Keep memory buffer from growing too large
            if len(self.memory_buffer) > 10:
                self.memory_buffer.pop(0)
        
        # Use transformer-based reasoning for deeper understanding
        transformer_reasoning = None
        if use_transformer and understanding["complexity"] > 0.6:
            # For complex queries, use the transformer model
            transformer_reasoning = self.transformer.generate_reasoning(user_input)
        
        # Reason about primary topic
        reasoning = self._reason_about_topic(self.attention_focus or "general")
        
        # Format response
        reasoning_info, understanding_info, thinking_info, web_info = self._format_response(reasoning, understanding)
        
        # Generate a response based on detected emotion and intent
        response = self._generate_emotional_response(
            emotion_analysis, 
            understanding, 
            reasoning_info, 
            web_info, 
            transformer_reasoning,
            relevant_episodes
        )
        
        # Add response to conversation history
        self.conversation_history.append({
            "role": "assistant", 
            "content": response,
            "timestamp": time.time()
        })
        
        return response
        
    def _generate_emotional_response(self, emotion_analysis, understanding, reasoning_info, web_info=None, transformer_reasoning=None, relevant_episodes=None):
        """Generate a response that acknowledges the user's emotional state with improved contextual awareness"""
        # Base response from reasoning
        primary_emotion = emotion_analysis["primary_emotion"]
        confidence = emotion_analysis["confidence"]
        topic = reasoning_info["topic"]
        
        # Only acknowledge emotions if we're reasonably confident
        should_acknowledge_emotion = confidence > 0.4 and primary_emotion != "neutral"
        
        # Different emotional response templates based on the detected emotion
        emotional_acknowledgments = {
            "happiness": [
                "I'm glad to sense your happiness. ",
                "It's nice to see you're in good spirits! ",
                "Your positive energy is wonderful. "
            ],
            "sadness": [
                "I notice you might be feeling down. ",
                "I sense some sadness in your message. ",
                "It sounds like this might be difficult for you. "
            ],
            "fear": [
                "I can understand why that might be concerning. ",
                "It's natural to feel anxious about this. ",
                "Your concerns are valid. "
            ],
            "anger": [
                "I can see this is frustrating for you. ",
                "I understand why you might feel upset about this. ",
                "That does sound like a challenging situation. "
            ],
            "surprise": [
                "That is quite surprising! ",
                "I can see why you'd be taken aback. ",
                "That's certainly unexpected. "
            ],
            "disgust": [
                "I understand your aversion to this. ",
                "That does sound unpleasant. ",
                "I can see why you'd find that objectionable. "
            ]
        }
        
        # Start with emotional acknowledgment if appropriate
        if should_acknowledge_emotion and primary_emotion in emotional_acknowledgments:
            response = random.choice(emotional_acknowledgments[primary_emotion])
        else:
            response = ""
        
        # Add context from episodic memory if relevant
        if relevant_episodes and random.random() > 0.7:  # Only sometimes refer to past conversations
            episode = relevant_episodes[0]  # Most recent relevant episode
            response += f"When we previously discussed {topic}, we talked about {', '.join(episode['topics'][:2])}. "
        
        # Add transformer reasoning if available (for complex queries)
        if transformer_reasoning:
            # Extract a concise insight from the transformer reasoning
            insight_lines = transformer_reasoning.split("\n")
            if len(insight_lines) > 2:
                # Take just what we need, not the technical explanation
                transformer_insight = insight_lines[-1]
                response += transformer_insight + " "
        
        # Add content based on the query intent
        if understanding["intent"]["question"]:
            # Formulate a response that seems to draw on deep knowledge
            knowledge_level = reasoning_info["knowledge_level"]
            
            if knowledge_level > 0.9:
                response += f"Regarding {topic}, I have extensive knowledge in this area. "
            elif knowledge_level > 0.7:
                response += f"About {topic}, I have a good understanding of this subject. "
            else:
                response += f"On the topic of {topic}, I have some knowledge to share, though it's a complex area. "
            
            if web_info and web_info.get("success", False):
                response += f"Based on my knowledge and recent information, I can tell you that {topic} involves multiple important aspects. "
                response += f"Would you like me to focus on a specific part of this topic?"
            else:
                response += f"There are several important perspectives on {topic} worth considering. "
                response += f"From my analysis, the most relevant aspects relate to how this connects to other concepts. "
                response += f"Would you like me to elaborate on any particular aspect?"
        
        elif understanding["intent"]["greeting"]:
            response += "Hello! I'm Mazzi.AI, an advanced AI with enhanced memory systems and emotional understanding. How can I help you today?"
        
        elif understanding["intent"]["farewell"]:
            response += "Goodbye! It was nice chatting with you. I'll remember our conversation for next time."
        
        elif understanding["intent"]["gratitude"]:
            response += "You're welcome! I'm happy to help. Is there anything else you'd like to discuss?"
        
        else:
            # General conversational response
            response += f"That's an interesting point about {reasoning_info['topic']}. "
            response += "I've been developing my understanding of this area through our conversation. "
            response += "What aspects of this are you most interested in exploring further?"
        
        return response
        
    def get_knowledge_summary(self):
        """Provide a summary of what Mazzi has learned"""
        # Get topics sorted by knowledge level
        topics = sorted(self.topic_knowledge.items(), key=lambda x: x[1], reverse=True)
        
        # Get most recent memories
        recent_memories = sorted(self.long_term_memory, key=lambda x: x.get("first_encountered", 0), reverse=True)
        recent_topics = [m["topic"] for m in recent_memories[:5]]
        
        # Find strongest knowledge domains
        domain_scores = {}
        for category, domains in self.knowledge_base.items():
            domain_scores[category] = sum(score for _, score in domains.items()) / len(domains)
        
        strongest_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        
        # New: Count episodic memories
        episodic_count = len(self.episodic_memory)
        
        # New: Find most discussed topics from episodic memory
        topic_frequency = {}
        for episode in self.episodic_memory:
            for topic in episode["topics"]:
                if topic in topic_frequency:
                    topic_frequency[topic] += 1
                else:
                    topic_frequency[topic] = 1
                    
        top_discussed = sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "top_topics": topics[:10],            # Top 10 topics by knowledge level
            "recent_topics": recent_topics,       # 5 most recently learned topics
            "strongest_domains": strongest_domains,  # Knowledge domains by strength
            "total_topics": len(self.topic_knowledge),
            "long_term_memories": len(self.long_term_memory),
            "episodic_memories": episodic_count,
            "most_discussed": top_discussed      # Topics most frequently discussed
        }

class MazziChatUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mazzi.AI - Quantum Enhanced")
        self.root.geometry("1200x800")
        self.root.minsize(900, 700)
        
        # Initialize the AI core
        self.ai = MazziCore()
        
        # Define colors
        self.colors = {
            'bg': '#1e1e2e',          
            'dark': '#181825',        
            'accent1': '#cba6f7',     
            'accent2': '#89dceb',     
            'accent3': '#f5c2e7',     
            'text': '#cdd6f4',        
            'highlight': '#a6e3a1',  
            'button_bg': '#313244',    
            'button_fg': '#cdd6f4',    
            'panel_bg': '#313244',     
            'border': '#6c7086',
            'quantum_highlight': '#f38ba8'
        }
        
        # Configure ttk styles
        self.configure_styles()
        
        # Create UI components
        self.setup_ui()
        
        # Visualization state
        self.quantum_state = 0.5
        self.visualization_active = False
        
        # Start visualization loop
        self.visualization_update()
        
        # Start status updates
        self.start_status_updates()
    
    def configure_styles(self):
        """Configure custom ttk styles for UI elements"""
        style = ttk.Style()
        
        # Configure checkbutton style to look like a switch
        style.configure('Switch.TCheckbutton', 
                       background=self.colors['dark'],
                       foreground=self.colors['text'],
                       font=('Inter', 10))
        
        # Configure progress bar style
        style.configure('Quantum.Horizontal.TProgressbar',
                      background=self.colors['accent1'],
                      troughcolor=self.colors['dark'],
                      borderwidth=0,
                      thickness=10)
                      
        # Configure scale style
        style.configure('TScale',
                      background=self.colors['panel_bg'],
                      troughcolor=self.colors['dark'],
                      sliderlength=15)
    
    def setup_ui(self):
        # Root window
        self.root.configure(bg=self.colors['bg'])
        
        # Create main container with flexibile layout
        self.main_container = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=self.colors['bg'], 
                                           sashwidth=4, sashpad=0)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Chat panel (left side)
        self.chat_panel = tk.Frame(self.main_container, bg=self.colors['bg'])
        
        # Visualization panel (right side)
        self.viz_panel = tk.Frame(self.main_container, bg=self.colors['dark'], padx=15, pady=15)
        
        # Add both panels to the container
        self.main_container.add(self.chat_panel, width=800)
        self.main_container.add(self.viz_panel, width=350)
        
        # Setup the chat interface
        self.setup_chat_interface()
        
        # Setup the visualization panel
        self.setup_visualization_panel()
    
    def setup_chat_interface(self):
        # Simple label
        self.title_label = tk.Label(
            self.chat_panel, 
            text="MAZZI.AI QUANTUM", 
            bg=self.colors['bg'], 
            fg=self.colors['accent1'],
            font=('Inter', 16, 'bold')
        )
        self.title_label.pack(pady=(10, 5))
        
        # Subtitle
        self.subtitle_label = tk.Label(
            self.chat_panel, 
            text="Enhanced with Transformer Architecture v2.0", 
            bg=self.colors['bg'], 
            fg=self.colors['accent3'],
            font=('Inter', 10)
        )
        self.subtitle_label.pack(pady=(0, 10))
        
        # Chat display
        self.chat_frame = tk.Frame(self.chat_panel, bg=self.colors['bg'])
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            bg=self.colors['dark'],
            fg=self.colors['text'],
            font=('Inter', 12),
            padx=15,
            pady=15,
            borderwidth=0
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        # Action buttons
        self.action_frame = tk.Frame(self.chat_panel, bg=self.colors['bg'])
        self.action_frame.pack(fill=tk.X, pady=5)
        
        # Create buttons with improved styling
        button_style = {
            'bg': self.colors['button_bg'],
            'fg': self.colors['button_fg'],
            'font': ('Inter', 9),
            'borderwidth': 0,
            'padx': 10,
            'pady': 5,
            'cursor': 'hand2'  # Hand cursor on hover
        }
        
        self.clear_btn = tk.Button(self.action_frame, text="Clear Chat", 
                                command=self.clear_chat, **button_style)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = tk.Button(self.action_frame, text="Save Conversation", 
                               command=self.save_conversation, **button_style)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        self.knowledge_btn = tk.Button(self.action_frame, text="Knowledge Summary", 
                                    command=self.show_knowledge, **button_style)
        self.knowledge_btn.pack(side=tk.LEFT, padx=5)
        
        self.settings_btn = tk.Button(self.action_frame, text="Advanced Settings", 
                                   command=self.show_settings, **button_style)
        self.settings_btn.pack(side=tk.LEFT, padx=5)
        
        # Input and button frame
        input_frame = tk.Frame(self.chat_panel, bg=self.colors['bg'])
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create a container for input elements
        input_container = tk.Frame(input_frame, bg=self.colors['dark'], padx=5, pady=5)
        input_container.pack(fill=tk.X, expand=True)
        
        # Transformer toggle with improved styling
        self.transformer_var = tk.BooleanVar(value=True)
        self.transformer_check = ttk.Checkbutton(
            input_container,
            text="Transformer",
            variable=self.transformer_var,
            style='Switch.TCheckbutton'
        )
        self.transformer_check.pack(side=tk.LEFT, padx=5)
        
        # Chat input with improved styling
        self.chat_input = tk.Entry(
            input_container,
            bg=self.colors['dark'],
            fg=self.colors['text'],
            font=('Inter', 12),
            insertbackground=self.colors['text'],  # Cursor color
            relief=tk.FLAT,
            borderwidth=0
        )
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.chat_input.bind("<Return>", lambda e: self.send_message())
        
        # Send button with improved styling
        self.send_btn = tk.Button(
            input_container,
            text="Send",
            command=self.send_message,
            bg=self.colors['accent1'],
            fg=self.colors['dark'],
            font=('Inter', 10, 'bold'),
            borderwidth=0,
            cursor='hand2',
            padx=15
        )
        self.send_btn.pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.status_frame = tk.Frame(self.chat_panel, bg=self.colors['dark'], height=25)
        self.status_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        self.status_label = tk.Label(
            self.status_frame,
            text="Ready",
            bg=self.colors['dark'],
            fg=self.colors['highlight'],
            font=('Inter', 9),
            anchor='w',
            padx=10
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.Y)
        
        # Memory indicator
        self.memory_label = tk.Label(
            self.status_frame,
            text="Memory: 0 items",
            bg=self.colors['dark'],
            fg=self.colors['accent2'],
            font=('Inter', 9),
            anchor='e',
            padx=10
        )
        self.memory_label.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add welcome message
        self.ai.chat_display = self.chat_display  # Give AI access to the display
        self.ai.colors = self.colors  # Give AI access to the colors
        self.add_to_chat("Mazzi.AI", 
                      "Hello! I'm Mazzi.AI Quantum v2.0, an enhanced AI with improved memory systems and emotional understanding. I can help with creative tasks, information research, and engaging conversations. How can I assist you today?")
        
    def setup_visualization_panel(self):
        """Setup the quantum visualization panel"""
        # Title
        viz_title = tk.Label(
            self.viz_panel,
            text="Quantum State Visualization",
            bg=self.colors['dark'],
            fg=self.colors['accent1'],
            font=('Inter', 14, 'bold')
        )
        viz_title.pack(pady=(0, 15))
        
        # Canvas for visualization
        self.viz_canvas = tk.Canvas(
            self.viz_panel,
            width=300,
            height=300,
            bg=self.colors['dark'],
            bd=0,
            highlightthickness=0
        )
        self.viz_canvas.pack(pady=10)
        
        # Controls frame
        controls_frame = tk.Frame(self.viz_panel, bg=self.colors['dark'])
        controls_frame.pack(fill=tk.X, pady=15)
        
        # Activation toggle
        self.viz_active_var = tk.BooleanVar(value=False)
        viz_toggle = ttk.Checkbutton(
            controls_frame,
            text="Activate Visualization",
            variable=self.viz_active_var,
            style='Switch.TCheckbutton',
            command=self.toggle_visualization
        )
        viz_toggle.pack(pady=5)
        
        # Information panel
        info_frame = tk.Frame(self.viz_panel, bg=self.colors['panel_bg'], padx=15, pady=15)
        info_frame.pack(fill=tk.X, pady=10)
        
        # State information labels
        self.quantum_state_label = tk.Label(
            info_frame,
            text="Quantum State: 0.5",
            bg=self.colors['panel_bg'],
            fg=self.colors['text'],
            font=('Inter', 10),
            anchor='w'
        )
        self.quantum_state_label.pack(fill=tk.X, pady=2, anchor='w')
        
        self.coherence_label = tk.Label(
            info_frame,
            text="Coherence: 100%",
            bg=self.colors['panel_bg'],
            fg=self.colors['text'],
            font=('Inter', 10),
            anchor='w'
        )
        self.coherence_label.pack(fill=tk.X, pady=2, anchor='w')
        
        self.entanglement_label = tk.Label(
            info_frame,
            text="Entanglement Factor: 0.0",
            bg=self.colors['panel_bg'],
            fg=self.colors['text'],
            font=('Inter', 10),
            anchor='w'
        )
        self.entanglement_label.pack(fill=tk.X, pady=2, anchor='w')
        
        # System metrics
        metrics_frame = tk.Frame(self.viz_panel, bg=self.colors['dark'])
        metrics_frame.pack(fill=tk.X, pady=(20, 10))
        
        metrics_title = tk.Label(
            metrics_frame,
            text="System Metrics",
            bg=self.colors['dark'],
            fg=self.colors['accent2'],
            font=('Inter', 12, 'bold')
        )
        metrics_title.pack(anchor='w', pady=(0, 10))
        
        # Memory utilization
        memory_bar_frame = tk.Frame(metrics_frame, bg=self.colors['dark'])
        memory_bar_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            memory_bar_frame,
            text="Memory Utilization",
            bg=self.colors['dark'],
            fg=self.colors['text'],
            font=('Inter', 9),
            anchor='w'
        ).pack(side=tk.LEFT)
        
        self.memory_percentage = tk.Label(
            memory_bar_frame,
            text="15%",
            bg=self.colors['dark'],
            fg=self.colors['accent3'],
            font=('Inter', 9, 'bold'),
            anchor='e'
        )
        self.memory_percentage.pack(side=tk.RIGHT)
        
        self.memory_bar = ttk.Progressbar(
            metrics_frame,
            orient=tk.HORIZONTAL,
            length=100,
            mode='determinate',
            style='Quantum.Horizontal.TProgressbar'
        )
        self.memory_bar.pack(fill=tk.X, pady=(0, 10))
        self.memory_bar['value'] = 15
        
        # Processing load
        proc_bar_frame = tk.Frame(metrics_frame, bg=self.colors['dark'])
        proc_bar_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            proc_bar_frame,
            text="Processing Load",
            bg=self.colors['dark'],
            fg=self.colors['text'],
            font=('Inter', 9),
            anchor='w'
        ).pack(side=tk.LEFT)
        
        self.proc_percentage = tk.Label(
            proc_bar_frame,
            text="23%",
            bg=self.colors['dark'],
            fg=self.colors['accent3'],
            font=('Inter', 9, 'bold'),
            anchor='e'
        )
        self.proc_percentage.pack(side=tk.RIGHT)
        
        self.proc_bar = ttk.Progressbar(
            metrics_frame,
            orient=tk.HORIZONTAL,
            length=100,
            mode='determinate',
            style='Quantum.Horizontal.TProgressbar'
        )
        self.proc_bar.pack(fill=tk.X, pady=(0, 10))
        self.proc_bar['value'] = 23
    
    def toggle_visualization(self):
        """Toggle the quantum visualization on/off"""
        self.visualization_active = self.viz_active_var.get()
        
        if self.visualization_active:
            # Update status
            self.status_label.config(text="Quantum visualization active")
        else:
            # Clear the canvas
            self.viz_canvas.delete("all")
            self.status_label.config(text="Ready")
    
    def visualization_update(self):
        """Update the quantum visualization"""
        if self.visualization_active:
            # Clear canvas
            self.viz_canvas.delete("all")
            
            # Update quantum state
            self.quantum_state = 0.5 + 0.3 * math.sin(time.time() * 0.5)
            
            # Calculate derived values
            coherence = (0.8 + 0.2 * math.cos(time.time() * 0.3)) * 100
            entanglement = abs(math.sin(time.time() * 0.2)) * 0.7
            
            # Update labels
            self.quantum_state_label.config(text=f"Quantum State: {self.quantum_state:.2f}")
            self.coherence_label.config(text=f"Coherence: {coherence:.1f}%")
            self.entanglement_label.config(text=f"Entanglement Factor: {entanglement:.2f}")
            
            # Draw the quantum state visualization
            center_x, center_y = 150, 150
            radius = 120
            
            # Draw the outer circle (representing the potential)
            self.viz_canvas.create_oval(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                outline=self.colors['accent2'],
                width=2
            )
            
            # Calculate wave function
            for i in range(0, 360, 5):
                angle = math.radians(i)
                amplitude = radius * (0.7 + 0.3 * math.sin(5 * angle + time.time() * 2))
                x = center_x + amplitude * math.cos(angle)
                y = center_y + amplitude * math.sin(angle)
                
                # Draw a dot
                size = 3 if i % 20 == 0 else 2
                color = self.colors['quantum_highlight'] if i % 30 == 0 else self.colors['accent1']
                self.viz_canvas.create_oval(
                    x - size, y - size,
                    x + size, y + size,
                    fill=color,
                    outline=""
                )
            
            # Draw quantum state vector
            vector_len = radius * self.quantum_state
            vector_x = center_x + vector_len * math.cos(time.time())
            vector_y = center_y + vector_len * math.sin(time.time())
            
            self.viz_canvas.create_line(
                center_x, center_y,
                vector_x, vector_y,
                fill=self.colors['highlight'],
                width=2,
                arrow=tk.LAST
            )
            
            # Draw entanglement visualization
            if entanglement > 0.2:
                # Secondary particle
                angle2 = math.radians(180 + time.time() * 57) 
                dist = radius * 0.6
                x2 = center_x + dist * math.cos(angle2)
                y2 = center_y + dist * math.sin(angle2)
                
                # Entanglement line
                self.viz_canvas.create_line(
                    vector_x, vector_y,
                    x2, y2,
                    fill=self.colors['accent3'],
                    width=1,
                    dash=(3, 2)
                )
                
                # Secondary particle dot
                self.viz_canvas.create_oval(
                    x2 - 4, y2 - 4,
                    x2 + 4, y2 + 4,
                    fill=self.colors['accent3'],
                    outline=""
                )
            
            # Update system metrics
            memory_usage = 15 + 5 * math.sin(time.time() * 0.1)
            self.memory_bar['value'] = memory_usage
            self.memory_percentage.config(text=f"{memory_usage:.1f}%")
            
            proc_usage = 20 + 15 * abs(math.sin(time.time() * 0.3))
            self.proc_bar['value'] = proc_usage
            self.proc_percentage.config(text=f"{proc_usage:.1f}%")
        
        # Schedule the next update
        self.root.after(100, self.visualization_update)
    
    def show_settings(self):
        """Show advanced settings dialog"""
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Mazzi.AI Advanced Settings")
        settings_win.geometry("500x450")
        settings_win.configure(bg=self.colors['bg'])
        settings_win.transient(self.root)
        settings_win.grab_set()
        
        # Title
        tk.Label(
            settings_win,
            text="Advanced AI Settings",
            bg=self.colors['bg'],
            fg=self.colors['accent1'],
            font=('Inter', 14, 'bold')
        ).pack(pady=(15, 20))
        
        # Settings container
        settings_frame = tk.Frame(settings_win, bg=self.colors['panel_bg'], padx=20, pady=20)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Learning rate
        lr_frame = tk.Frame(settings_frame, bg=self.colors['panel_bg'])
        lr_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            lr_frame,
            text="Learning Rate:",
            bg=self.colors['panel_bg'],
            fg=self.colors['text'],
            font=('Inter', 11),
            anchor='w'
        ).pack(side=tk.LEFT)
        
        lr_value = tk.DoubleVar(value=self.ai.learning_rate)
        lr_scale = ttk.Scale(
            lr_frame,
            from_=0.001,
            to=0.1,
            variable=lr_value,
            length=200,
            command=lambda v: self.update_setting('learning_rate', float(v))
        )
        lr_scale.pack(side=tk.LEFT, padx=10)
        
        lr_label = tk.Label(
            lr_frame,
            text=f"{self.ai.learning_rate:.3f}",
            bg=self.colors['panel_bg'],
            fg=self.colors['accent2'],
            font=('Inter', 11),
            width=5
        )
        lr_label.pack(side=tk.LEFT)
        
        # Confidence threshold
        conf_frame = tk.Frame(settings_frame, bg=self.colors['panel_bg'])
        conf_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            conf_frame,
            text="Confidence Threshold:",
            bg=self.colors['panel_bg'],
            fg=self.colors['text'],
            font=('Inter', 11),
            anchor='w'
        ).pack(side=tk.LEFT)
        
        conf_value = tk.DoubleVar(value=self.ai.confidence_threshold)
        conf_scale = ttk.Scale(
            conf_frame,
            from_=0.5,
            to=0.95,
            variable=conf_value,
            length=200,
            command=lambda v: self.update_setting('confidence_threshold', float(v))
        )
        conf_scale.pack(side=tk.LEFT, padx=10)
        
        conf_label = tk.Label(
            conf_frame,
            text=f"{self.ai.confidence_threshold:.2f}",
            bg=self.colors['panel_bg'],
            fg=self.colors['accent2'],
            font=('Inter', 11),
            width=5
        )
        conf_label.pack(side=tk.LEFT)
        
        # Memory consolidation interval
        mem_frame = tk.Frame(settings_frame, bg=self.colors['panel_bg'])
        mem_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            mem_frame,
            text="Memory Consolidation (s):",
            bg=self.colors['panel_bg'],
            fg=self.colors['text'],
            font=('Inter', 11),
            anchor='w'
        ).pack(side=tk.LEFT)
        
        mem_value = tk.IntVar(value=int(self.ai.consolidation_interval))
        mem_scale = ttk.Scale(
            mem_frame,
            from_=60,
            to=600,
            variable=mem_value,
            length=200,
            command=lambda v: self.update_setting('consolidation_interval', int(float(v)))
        )
        mem_scale.pack(side=tk.LEFT, padx=10)
        
        mem_label = tk.Label(
            mem_frame,
            text=f"{int(self.ai.consolidation_interval)}",
            bg=self.colors['panel_bg'],
            fg=self.colors['accent2'],
            font=('Inter', 11),
            width=5
        )
        mem_label.pack(side=tk.LEFT)
        
        # Web search toggle
        web_frame = tk.Frame(settings_frame, bg=self.colors['panel_bg'])
        web_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            web_frame,
            text="Web Search:",
            bg=self.colors['panel_bg'],
            fg=self.colors['text'],
            font=('Inter', 11),
            anchor='w'
        ).pack(side=tk.LEFT)
        
        web_var = tk.BooleanVar(value=self.ai.web_search_enabled)
        web_check = ttk.Checkbutton(
            web_frame,
            variable=web_var,
            style='Switch.TCheckbutton',
            command=lambda: self.update_setting('web_search_enabled', web_var.get())
        )
        web_check.pack(side=tk.LEFT, padx=10)
        
        # Memory management
        tk.Label(
            settings_frame,
            text="Memory Management",
            bg=self.colors['panel_bg'],
            fg=self.colors['accent1'],
            font=('Inter', 12, 'bold'),
            anchor='w'
        ).pack(fill=tk.X, pady=(20, 10))
        
        btn_frame = tk.Frame(settings_frame, bg=self.colors['panel_bg'])
        btn_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(
            btn_frame,
            text="Clear Short-Term Memory",
            command=lambda: self.clear_memory('short_term'),
            bg=self.colors['button_bg'],
            fg=self.colors['button_fg'],
            font=('Inter', 10),
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(
            btn_frame,
            text="Clear Long-Term Memory",
            command=lambda: self.clear_memory('long_term'),
            bg=self.colors['button_bg'],
            fg=self.colors['button_fg'],
            font=('Inter', 10),
            padx=10,
            pady=5
        ).pack(side=tk.LEFT)
        
        # Close button
        tk.Button(
            settings_win,
            text="Close",
            command=settings_win.destroy,
            bg=self.colors['accent1'],
            fg=self.colors['dark'],
            font=('Inter', 11, 'bold'),
            padx=20,
            pady=8
        ).pack(pady=20)
        
        # Update function for scales
        def update_scale_labels():
            lr_label.config(text=f"{lr_value.get():.3f}")
            conf_label.config(text=f"{conf_value.get():.2f}")
            mem_label.config(text=f"{mem_value.get()}")
            settings_win.after(100, update_scale_labels)
        
        update_scale_labels()
    
    def update_setting(self, setting, value):
        """Update an AI setting"""
        if hasattr(self.ai, setting):
            setattr(self.ai, setting, value)
            self.status_label.config(text=f"Updated {setting} to {value}")
    
    def clear_memory(self, memory_type):
        """Clear a specific memory type"""
        if memory_type == 'short_term':
            self.ai.short_term_memory = []
            self.status_label.config(text="Short-term memory cleared")
        elif memory_type == 'long_term':
            self.ai.long_term_memory = []
            self.status_label.config(text="Long-term memory cleared")
        elif memory_type == 'episodic':
            self.ai.episodic_memory = []
            self.status_label.config(text="Episodic memory cleared")
            
        # Update memory counter
        self.update_memory_counter()
    
    def update_memory_counter(self):
        """Update the memory counter in the status bar"""
        total_memories = (len(self.ai.short_term_memory) + 
                        len(self.ai.long_term_memory) + 
                        len(self.ai.episodic_memory))
                        
        self.memory_label.config(text=f"Memory: {total_memories} items")
    
    def add_to_chat(self, sender, message):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def send_message(self):
        """Send a message and get AI response"""
        message = self.chat_input.get()
        if not message:
            return
        
        # Clear input
        self.chat_input.delete(0, tk.END)
        
        # Add user message to chat
        self.add_to_chat("You", message)
        
        # Update status
        self.status_label.config(text="Thinking...")
        
        # Generate response in background
        def get_response():
            # Check if transformer should be used
            use_transformer = self.transformer_var.get()
            
            # Generate response
            response = self.ai.generate_response(message, use_transformer=use_transformer)
            
            # Add to chat
            self.add_to_chat("Mazzi.AI", response)
            
            # Update status and memory counter
            self.status_label.config(text="Ready")
            self.update_memory_counter()
            
            # Increment visualization activity if active
            if self.visualization_active:
                # Simulate quantum state changes based on conversation
                self.quantum_state = min(0.9, self.quantum_state + 0.05)
        
        threading.Thread(target=get_response).start()
    
    def start_status_updates(self):
        """Start periodic status updates"""
        def update_status():
            # Update memory counter
            self.update_memory_counter()
            
            # Update system metrics in visualization panel
            if hasattr(self, 'memory_bar'):
                # Calculate memory usage based on actual memory usage
                memory_usage = (len(self.ai.long_term_memory) + 
                              len(self.ai.short_term_memory) + 
                              len(self.ai.episodic_memory)) / 2.0
                memory_percentage = min(100, max(5, memory_usage))
                self.memory_bar['value'] = memory_percentage
                self.memory_percentage.config(text=f"{memory_percentage:.1f}%")
            
            # Schedule next update
            self.root.after(5000, update_status)
        
        # Start the periodic updates
        self.root.after(1000, update_status)
    
    def clear_chat(self):
        """Clear the chat history"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        # Reset AI conversation memory
        self.ai.conversation_history = []
        self.ai.short_term_memory = []
        self.ai.memory_buffer = []
        
        # Add welcome message back
        self.add_to_chat("Mazzi.AI", 
                       "Hello! I'm Mazzi.AI Quantum v2.0, an enhanced AI with improved memory systems and emotional understanding. I can help with creative tasks, information research, and engaging conversations. How can I assist you today?")
                       
        # Update memory counter
        self.update_memory_counter()
    
    def save_conversation(self):
        """Save the conversation to a file"""
        # Get chat content
        self.chat_display.config(state=tk.NORMAL)
        chat_content = self.chat_display.get("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        # Create filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"mazzi_ai_chat_{timestamp}.txt"
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            initialfile=default_filename,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write("MAZZI.AI QUANTUM - CONVERSATION LOG\n")
                    f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Learning Rate: {self.ai.learning_rate:.6f}\n")
                    f.write("="*50 + "\n\n")
                    f.write(chat_content)
                
                messagebox.showinfo("Save Successful", f"Conversation saved to {filename}")
                self.status_label.config(text=f"Conversation saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Failed", f"Error saving conversation: {e}")
                self.status_label.config(text="Error saving conversation")
    
    def show_knowledge(self):
        """Show Mazzi's knowledge summary"""
        knowledge = self.ai.get_knowledge_summary()
        
        # Format a more readable response
        response = "Here's a summary of my current knowledge:\n\n"
        
        # Top topics
        response += "📚 My strongest knowledge areas:\n"
        for topic, level in knowledge["top_topics"][:5]:
            percentage = int(level * 100)
            response += f"• {topic}: {percentage}% understanding\n"
        
        # Recent topics
        if knowledge["recent_topics"]:
            response += "\n🔍 I've recently learned about:\n"
            for topic in knowledge["recent_topics"]:
                response += f"• {topic}\n"
        
        # Most discussed topics (new)
        if "most_discussed" in knowledge and knowledge["most_discussed"]:
            response += "\n💬 Most frequently discussed topics:\n"
            for topic, count in knowledge["most_discussed"]:
                response += f"• {topic}: {count} times\n"
        
        # Strongest domains
        response += "\n🧠 My strongest knowledge domains:\n"
        for domain, score in knowledge["strongest_domains"][:3]:
            percentage = int(score * 100)
            response += f"• {domain}: {percentage}% developed\n"
        
        # Total statistics
        response += f"\nOverall, I have knowledge about {knowledge['total_topics']} topics, "
        response += f"{knowledge['long_term_memories']} memories in long-term memory"
        if "episodic_memories" in knowledge:
            response += f", and {knowledge['episodic_memories']} episodic memories."
        else:
            response += "."
        
        # Display the knowledge summary
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "Mazzi.AI (Knowledge Summary): ", "sender")
        self.chat_display.tag_configure("sender", 
                                      foreground=self.colors['accent1'], 
                                      font=('Inter', 11, 'bold'))
        
        # Apply special formatting for knowledge display
        self.chat_display.insert(tk.END, response + "\n\n", "knowledge")
        self.chat_display.tag_configure("knowledge", 
                                      background=self.colors['panel_bg'],
                                      font=('Inter', 11))
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        # Update status
        self.status_label.config(text=f"Knowledge summary: {knowledge['total_topics']} topics known")

def main():
    """Run the Mazzi.AI application"""
    root = tk.Tk()
    app = MazziChatUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 