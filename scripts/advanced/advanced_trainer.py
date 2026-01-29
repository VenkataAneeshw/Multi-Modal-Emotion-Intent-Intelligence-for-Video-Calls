import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import logging
from tqdm import tqdm
import wandb
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)

class AdvancedTrainer:
    """
    Advanced training framework with mixed precision, distributed training,
    and modern optimization techniques.
    """

    def __init__(self, model, train_dataset, val_dataset, config):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Distributed training setup
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

        self.is_distributed = self.world_size > 1
        self.is_main_process = self.rank == 0

        if self.is_distributed:
            self._setup_distributed()

        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None

        # Optimizer with advanced scheduling
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Loss functions with label smoothing
        self.criterion = {
            'emotion': nn.CrossEntropyLoss(label_smoothing=0.1),
            'intent': nn.CrossEntropyLoss(label_smoothing=0.1),
            'engagement': self._create_regression_loss(),
            'confidence': self._create_regression_loss(),
            'contrastive': nn.CrossEntropyLoss()
        }

        # Weights for multi-task loss
        self.task_weights = config.task_weights

        # Initialize wandb for main process
        if self.is_main_process and config.use_wandb:
            wandb.init(project="emotia-training", config=config.__dict__)

    def _setup_distributed(self):
        """Setup distributed training"""
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )

        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _create_optimizer(self):
        """Create advanced optimizer"""
        if self.config.optimizer == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999)
            )
        elif self.config.optimizer == 'lion':
            # LION optimizer (more memory efficient)
            from lion_pytorch import Lion
            optimizer = Lion(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )

        return optimizer

    def _create_scheduler(self):
        """Create advanced learning rate scheduler"""
        if self.config.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler == 'one_cycle':
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr,
                epochs=self.config.epochs,
                steps_per_epoch=len(self.train_dataset) // (self.config.batch_size * self.world_size),
                pct_start=0.3,
                anneal_strategy='cos'
            )
        else:
            scheduler = None

        return scheduler

    def _create_regression_loss(self):
        """Create regression loss with uncertainty"""
        def uncertainty_loss(pred_mean, pred_var, target):
            # Negative log likelihood for Gaussian distribution
            loss = 0.5 * torch.log(pred_var) + 0.5 * (target - pred_mean)**2 / pred_var
            return loss.mean()

        return uncertainty_loss

    def train_epoch(self, epoch):
        """Train for one epoch with advanced techniques"""
        self.model.train()

        if self.is_distributed:
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
            dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True
            )

        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}") if self.is_main_process else dataloader

        for batch in progress_bar:
            # Move to device
            batch = {k: v.cuda(self.local_rank) if torch.is_tensor(v) else v for k, v in batch.items()}

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.scaler:
                with autocast():
                    outputs = self.model(**batch)
                    loss = self._compute_loss(outputs, batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**batch)
                loss = self._compute_loss(outputs, batch)
                loss.backward()
                self.optimizer.step()

            # Update scheduler (for OneCycleLR)
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            if self.is_main_process:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches

        # Step scheduler (for CosineAnnealingLR)
        if isinstance(self.scheduler, CosineAnnealingLR):
            self.scheduler.step()

        return avg_loss

    def _compute_loss(self, outputs, batch):
        """Compute multi-task loss with uncertainty"""
        total_loss = 0

        # Emotion classification
        if 'emotion_logits' in outputs and 'emotion' in batch:
            emotion_loss = self.criterion['emotion'](outputs['emotion_logits'], batch['emotion'])
            total_loss += self.task_weights['emotion'] * emotion_loss

        # Intent classification
        if 'intent_logits' in outputs and 'intent' in batch:
            intent_loss = self.criterion['intent'](outputs['intent_logits'], batch['intent'])
            total_loss += self.task_weights['intent'] * intent_loss

        # Engagement regression with uncertainty
        if 'engagement_mean' in outputs and 'engagement_var' in outputs and 'engagement' in batch:
            engagement_loss = self.criterion['engagement'](
                outputs['engagement_mean'], outputs['engagement_var'], batch['engagement']
            )
            total_loss += self.task_weights['engagement'] * engagement_loss

        # Confidence regression with uncertainty
        if 'confidence_mean' in outputs and 'confidence_var' in outputs and 'confidence' in batch:
            confidence_loss = self.criterion['confidence'](
                outputs['confidence_mean'], outputs['confidence_var'], batch['confidence']
            )
            total_loss += self.task_weights['confidence'] * confidence_loss

        # Contrastive loss for multi-modal alignment
        if hasattr(self.model, 'contrastive_loss') and 'embeddings' in outputs:
            contrastive_loss = self.model.contrastive_loss(outputs['embeddings'])
            total_loss += self.config.contrastive_weight * contrastive_loss

        return total_loss

    def validate(self, epoch):
        """Validation with comprehensive metrics"""
        self.model.eval()

        if self.is_distributed:
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
            dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )

        total_loss = 0
        num_batches = 0

        all_emotion_preds = []
        all_emotion_labels = []
        all_intent_preds = []
        all_intent_labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.cuda(self.local_rank) if torch.is_tensor(v) else v for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = self._compute_loss(outputs, batch)

                total_loss += loss.item()
                num_batches += 1

                # Collect predictions for metrics
                if 'emotion_logits' in outputs:
                    all_emotion_preds.extend(outputs['emotion_logits'].argmax(dim=1).cpu().numpy())
                    all_emotion_labels.extend(batch['emotion'].cpu().numpy())

                if 'intent_logits' in outputs:
                    all_intent_preds.extend(outputs['intent_logits'].argmax(dim=1).cpu().numpy())
                    all_intent_labels.extend(batch['intent'].cpu().numpy())

        avg_loss = total_loss / num_batches

        # Compute metrics
        metrics = self._compute_metrics(all_emotion_preds, all_emotion_labels,
                                      all_intent_preds, all_intent_labels)

        return avg_loss, metrics

    def _compute_metrics(self, emotion_preds, emotion_labels, intent_preds, intent_labels):
        """Compute comprehensive evaluation metrics"""
        from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

        metrics = {}

        if emotion_preds and emotion_labels:
            metrics.update({
                'emotion_accuracy': accuracy_score(emotion_labels, emotion_preds),
                'emotion_f1_macro': f1_score(emotion_labels, emotion_preds, average='macro'),
                'emotion_f1_weighted': f1_score(emotion_labels, emotion_preds, average='weighted'),
            })

        if intent_preds and intent_labels:
            metrics.update({
                'intent_accuracy': accuracy_score(intent_labels, intent_preds),
                'intent_f1_macro': f1_score(intent_labels, intent_preds, average='macro'),
                'intent_f1_weighted': f1_score(intent_labels, intent_preds, average='weighted'),
            })

        return metrics

    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Train epoch
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_metrics = self.validate(epoch)

            # Log metrics
            if self.is_main_process:
                logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
                for metric_name, metric_value in val_metrics.items():
                    logger.info(f"{metric_name}: {metric_value:.4f}")

                # Wandb logging
                if self.config.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        **val_metrics,
                        'lr': self.optimizer.param_groups[0]['lr']
                    })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if self.is_main_process:
                    self.save_checkpoint(epoch, val_loss, val_metrics)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config.patience:
                logger.info("Early stopping triggered")
                break

        # Final cleanup
        if self.is_distributed:
            dist.destroy_process_group()

    def save_checkpoint(self, epoch, val_loss, val_metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'config': self.config
        }

        checkpoint_path = f"{self.config.checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    @staticmethod
    def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        return checkpoint['epoch'], checkpoint['val_loss'], checkpoint['val_metrics']