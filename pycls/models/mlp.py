# mlp.py

# import torch
# import torch.nn as nn

# class MLPClassifierTorch(nn.Module):
#     def __init__(self, input_dim, num_classes, use_dropout=True):
#         super(MLPClassifierTorch, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 64)
#         self.dropout1 = nn.Dropout(p=0.5) if use_dropout else nn.Identity()
#         self.fc2 = nn.Linear(64, 8)
#         self.dropout2 = nn.Dropout(p=0.5) if use_dropout else nn.Identity()
#         self.fc3 = nn.Linear(8, num_classes)  # num_classes should be 2 for binary classification
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)  # Output logits
#         return x


# mlp.py

import torch
import torch.nn as nn

class MLPClassifierTorch(nn.Module):
    def __init__(self, input_dim, use_dropout=True):
        super(MLPClassifierTorch, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(p=0.5) if use_dropout else nn.Identity()
        self.fc2 = nn.Linear(64, 8)
        self.dropout2 = nn.Dropout(p=0.5) if use_dropout else nn.Identity()
        self.fc3 = nn.Linear(8, 1)  # Single output unit for binary classification
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# # Example instantiation, loss, and optimizer setup
# if __name__ == "__main__":
#     model = MLPClassifierTorch(input_dim=30)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     # Example input tensor
#     x = torch.randn(1, 30)  # Batch size of 1, input dimension of 30
#     output = model(x)
#     print("Model output:", output)
