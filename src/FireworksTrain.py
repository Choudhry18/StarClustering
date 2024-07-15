# Define the objective function for Fireworks Algorithm
#! /usr/bin/env python

import os
import sys
import time
import numpy as np
import pickle
sys.path.insert(0, './src/utils')
sys.path.insert(0, './model')

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_du
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from starcnet import Net
import torch.nn.init as init


def objective(hyperparameters):
    dropout_rate1, dropout_rate2, learning_rate = hyperparameters
    model = Net(dropout_rate1, dropout_rate2, learning_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loader, val_loader = get_data()
    
    # Training
    model.train()
    for epoch in range(10):  # Increase epochs for real optimization
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    accuracy = correct / total
    return -accuracy  # We want to maximize accuracy, hence minimize -accuracy

# Fireworks Algorithm (barebones implementation)
def fireworks_algorithm(bounds, n_particles, max_iter):
    lower_bounds, upper_bounds = bounds
    dimensions = len(lower_bounds)
    particles = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(n_particles, dimensions))
    best_particle = None
    best_fitness = np.inf
    
    for t in range(max_iter):
        for i, particle in enumerate(particles):
            fitness = objective(particle)
            if fitness < best_fitness:
                best_fitness = fitness
                best_particle = particle
            
            # Firework explosion
            for j in range(n_particles):
                if fitness < objective(particles[j]):
                    particles[j] = particles[j] + np.random.uniform(low=-1, high=1, size=dimensions) * np.abs(particles[j] - particle)
        
    return best_particle, best_fitness

# Set bounds for the hyperparameters
hyperparam_bounds = [[0.1, 0.5], [0.1, 0.5], [1e-4, 1e-2]]  # Example bounds for dropout rates and learning rate

# Run Fireworks Algorithm
best_params, best_accuracy = fireworks_algorithm(hyperparam_bounds, n_particles=10, max_iter=5)

print(f"Best Hyperparameters: {best_params}")
print(f"Best Validation Accuracy: {-best_accuracy}")

# Train the final model with the best hyperparameters
dropout_rate1, dropout_rate2, learning_rate = best_params
model = Net(dropout_rate1, dropout_rate2, learning_rate)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader, val_loader = get_data()

# Training the final model
model.train()
for epoch in range(20):  # Increase epochs for real training
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Evaluate the final model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

accuracy = correct / total
print(f"Final Model Test Accuracy: {accuracy}")
