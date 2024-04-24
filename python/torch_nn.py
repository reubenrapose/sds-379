import torch
import torch.nn as nn
import numpy as np
import sys
from matplotlib import pyplot
import os

# Define a NN class for 2 hidden layers
class NeuralNet(nn.Module):
    def __init__(self, hidden_size, output_size=1,input_size=1):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.Tanh()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.Tanh()
        self.l3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out

def main():
    file_prefix = "torch_pde4"
    print(f"Run begin for {file_prefix}\n")
    run_model(file_prefix)
    # test_model(file_prefix)
    print("Complete")

def run_model(file_prefix):
    # Organize outputs
    file_name = get_unique_file_name("outputs/" + file_prefix + ".txt")
    with open(file_name, 'w') as file:
        # Redirect stdout to the file
        sys.stdout = file
        torch_nn_pde1(file_prefix)

    # Reset stdout to the default (console)
    sys.stdout = sys.__stdout__

def torch_nn_ode2(file_prefix):
    # Create the criterion that will be used for the DE part of the loss
    criterion = nn.MSELoss()

    # Time vector that will be used as input of our NN
    z = (np.array([np.random.rand()
                      for _ in range(49)])).astype(float)
    z = np.insert(z, 0, 0.0) # ensure t=0 at index 0 in sample for initial condition
    z = torch.from_numpy(z).reshape(len(z), 1).float()
    z.requires_grad_(True) # For differentiation

    hidden_size = 10
    model = NeuralNet(hidden_size=10, input_size=1) # 10x2 Network

    # Loss and optimizer
    learning_rate = .03
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Number of epochs
    max_iters = 30000000
    tol = 1e-4
    upper_limit = 500

    print(f"{hidden_size}x{2} Network to soln to x_tt + x = 0, x(0) = 1, x'(0)=2:\n\n")

    for iter in range(1,max_iters+1):
        printer = iter == 1 or iter % 10000 == 0

        # Forward pass
        y_pred = model(z)

        # Calculate the derivative of the forward pass w.r.t. the input (t)
        dxdt = torch.autograd.grad(y_pred,
                                    z,
                                    grad_outputs=torch.ones_like(y_pred),
                                    create_graph=True,
                                    retain_graph=True)[0]
        d2xdt2 = torch.autograd.grad(dxdt,
                                      z,
                                      grad_outputs=torch.ones_like(dxdt),
                                      create_graph=True,
                                      retain_graph=True)[0]

        # Define the differential equation and calculate the loss
        loss_DE = criterion(d2xdt2 + z, torch.zeros_like(z))

        # Define the initial condition loss
        loss_IC1 = abs(y_pred[0] - 1.0)
        loss_IC2 = abs(dxdt[0] - 2.0)

        loss = loss_DE + loss_IC1 + loss_IC2
        loss = loss[0]

        # Backward pass and weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if printer:
            print("Iter: ", iter, f"|| Loss: {loss:.10f}")
            if loss < tol:
                print("Converged.")
                break
            elif loss > upper_limit:
                print(f"Fitting err > {upper_limit}.\n")

    file_name = get_unique_file_name("outputs/"+file_prefix+".pth")
    torch.save(model, file_name)

def torch_nn_pde1(file_prefix):
    # Create the criterion that will be used for the DE part of the loss
    criterion = nn.MSELoss()

    # Time vector that will be used as input of our NN
    zpde = (np.array([[np.random.rand(), np.random.rand()]
                      for _ in range(25)])).astype(float)
    zpde = torch.from_numpy(zpde).reshape(len(zpde), 2).float()
    zpde.requires_grad_(True)
    zic = (np.array([[np.random.rand(), 0] for _ in range(10)])).astype(float)
    zic = torch.from_numpy(zic).reshape(len(zic), 2).float()
    zbc1 = (np.array([[0, np.random.rand()] for _ in range(10)])).astype(float)
    zbc1 = torch.from_numpy(zbc1).reshape(len(zbc1), 2).float()
    zbc2 = (np.array([[0, np.random.rand()] for _ in range(10)])).astype(float)
    zbc2 = torch.from_numpy(zbc2).reshape(len(zbc2), 2).float()

    # Instantiate one model with 50 neurons on the hidden layers
    hidden_size = 10
    model = NeuralNet(hidden_size=10, input_size=2)

    # Loss and optimizer
    learning_rate = .03
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Number of epochs
    max_iters = 30000000
    tol = 1e-4
    upper_limit = 500

    print(f"{hidden_size}x{2} Network to soln to u_t - 1/pi*u_xx = 0"
          f"u(x,0) = sin(pi*x), u(0,t) = u(1,t) = 0:\n\n")

    for iter in range(1,max_iters+1):
        printer = iter == 1 or iter % 10000 == 0

        # Forward pass
        y_pred_pde = model(zpde)
        y_pred_ic = model(zic)
        y_pred_bc1 = model(zbc1)
        y_pred_bc2 = model(zbc2)

        # Calculate the derivative of the forward pass w.r.t. the input (t)
        du = torch.autograd.grad(y_pred_pde,
                                    zpde,
                                    grad_outputs=torch.ones_like(y_pred_pde),
                                    create_graph=True,
                                    retain_graph=True)[0]
        du_dx = du[:,0] # u_xx
        d2u_dx2 = torch.autograd.grad(du_dx,
                                      zpde,
                                      grad_outputs=torch.ones_like(du_dx),
                                      create_graph=True,
                                      retain_graph=True)[0][:,0]
        du_dt = du[:,1] # u_t

        # Define the differential equation and calculate the loss
        loss_DE = criterion(du_dt - 1/torch.pi * d2u_dx2, torch.zeros_like(d2u_dx2))

        # Define the initial and boundary conditions losses
        loss_IC = criterion(y_pred_ic, torch.sin(torch.pi*zic))
        loss_BC1 = criterion(y_pred_bc1, torch.zeros_like(y_pred_bc1))
        loss_BC2 = criterion(y_pred_bc2, torch.zeros_like(y_pred_bc2))

        loss = loss_DE + loss_IC + loss_BC1 + loss_BC2

        # Backward pass and weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if printer:
            print("Iter: ", iter, f"|| Loss: {loss:.10f}")
            if loss < tol:
                print("Converged.")
                break
            elif loss > upper_limit:
                print(f"Fitting err > {upper_limit}.\n")

    file_name = get_unique_file_name("outputs/"+file_prefix+".pth")
    torch.save(model, file_name)

    # Define the loss function for the initial condition
def initial_condition_loss(y, target_value):
    return nn.MSELoss()(y, target_value)

# the following function plots true solution vs approximated solution
# the input must be altered for the specific case
def test_model(file_prefix):
    model = torch.load("outputs/" + file_prefix + ".pth")
    model.eval()

    z1 = np.arange(0, 1.01, .01)
    z2 = np.arange(0, 1.01, .01)
    z = torch.stack((torch.from_numpy(z1), torch.from_numpy(z2)), dim=1)

    fz = np.exp(-1*np.pi*z2) * np.sin # dependent on true solution
    yhat = model(z)
    yhat = yhat.detach().numpy()

    pyplot.scatter(z, fz, label="Actual")
    pyplot.scatter(z, yhat, label="Predicted")
    pyplot.title("Approximating soln to x_tt + x = 0, x(0) = 1, x'(0) = 2")
    pyplot.xlabel("x")
    pyplot.ylabel("f(x) vs N(x)")
    pyplot.legend()
    pyplot.show()


# This function is for testing convenience only: helps avoid overwriting previous runs
def get_unique_file_name(file_name):
    # Check if the file exists in the current working directory
    if os.path.exists(file_name):
        base_name, extension = os.path.splitext(file_name)
        base_name_no_num = base_name[:len(base_name)-1]
        counter = 1
        # Keep incrementing the counter until a unique file name is found
        while True:
            new_file_name = f"{base_name_no_num}{counter}{extension}"
            if not os.path.exists(new_file_name):
                return new_file_name
            counter += 1
    else:
        return file_name

if __name__ == "__main__":
    main()
