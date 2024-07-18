import timm
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F


class LitEfficientNet(L.LightningModule):
    def __init__(self):
        super().__init__()

        # needed by the TensorBoard logger to extract the model graph
        self.example_input_array = torch.rand(1, 3, 384, 384)
        
        # initialize the pretrained model
        self._prepare_model()
        
    def _prepare_model(self):
        # initialize a pretrained instance of the model
        self.model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True)
        
        # grab the number of input features to the classifier
        num_features = self.model.classifier.in_features
        
        # add a new binary classfier
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 1))
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = self(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, labels[:, None].half())
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        inputs, labels = valid_batch
        outputs = self(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, labels[:, None].half())
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)

