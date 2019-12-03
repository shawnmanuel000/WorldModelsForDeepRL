import torch
import argparse
import numpy as np
from torchvision import transforms
from models.mdrnn import MDRNN
from models.vae import VAE, LATENT_SIZE
from data.loaders import RolloutSequenceDataset, ROOT

parser = argparse.ArgumentParser(description='MDRNN Trainer')
parser.add_argument('--epochs', type=int, default=50, help='how many worker servers to connect to')
parser.add_argument('--iternum', type=int, default=0, help='which port to listen on (as a worker server)')
args = parser.parse_args()

SEQ_LEN = 32
BATCH_SIZE = 16
TRAIN_BUFFER = 30
TEST_BUFFER = 10
NUM_WORKERS = 0

def get_data_loaders(dataset_path=ROOT):
	transform = transforms.Lambda(lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
	dataset_train = RolloutSequenceDataset(SEQ_LEN, transform, dataset_path, train=True, buffer_size=TRAIN_BUFFER)
	dataset_test = RolloutSequenceDataset(SEQ_LEN, transform, dataset_path, train=False, buffer_size=TEST_BUFFER)
	train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
	test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
	return train_loader, test_loader

def train_loop(train_loader, vae, mdrnn):
	mdrnn.train()
	batch_losses = []
	train_loader.dataset.load_next_buffer()
	for states, actions, next_states, rewards, dones in train_loader:
		latents, next_latents = [vae.get_latents(x).transpose(1,0) for x in (states, next_states)]
		actions, rewards, dones = [x.transpose(1,0) for x in (actions, rewards, dones)]
		loss = mdrnn.optimize(latents, actions, next_latents, rewards, dones)
		batch_losses.append(loss)
	return np.sum(batch_losses) * BATCH_SIZE / len(train_loader.dataset)

def test_loop(test_loader, vae, mdrnn):
	mdrnn.eval()
	batch_losses = []
	test_loader.dataset.load_next_buffer()
	for states, actions, next_states, rewards, dones in test_loader:
		with torch.no_grad():
			latents, next_latents = [vae.get_latents(x).transpose(1,0) for x in (states, next_states)]
			actions, rewards, dones = [x.transpose(1,0) for x in (actions, rewards, dones)]
			loss = mdrnn.get_loss(latents, actions, next_latents, rewards, dones).item()
			batch_losses.append(loss)
	return np.sum(batch_losses) * BATCH_SIZE / len(test_loader.dataset)

def run(epochs=50, checkpoint_dirname="pytorch"):
	train_loader, test_loader = get_data_loaders()
	vae = VAE().load_model(checkpoint_dirname)
	mdrnn = MDRNN(load=False)
	ep_train_losses = []
	ep_test_losses = []
	for ep in range(epochs):
		train_loss = train_loop(train_loader, vae, mdrnn)
		test_loss = test_loop(test_loader, vae, mdrnn)
		ep_train_losses.append(train_loss)
		ep_test_losses.append(test_loss)
		mdrnn.schedule(test_loss)
		mdrnn.save_model(checkpoint_dirname, "latest")
		if ep_test_losses[-1] <= np.min(ep_test_losses): mdrnn.save_model(checkpoint_dirname)
		print(f"Ep: {ep+1} / {epochs}, Train: {ep_train_losses[-1]:.4f}, Test: {ep_test_losses[-1]:.4f}")
		with open(f"logs/mdrnn/logs.txt", "a+") as f:
			f.write(f"Ep: {ep}, Train: {ep_train_losses[-1]}, Test: {ep_test_losses[-1]}\n")
		
if __name__ == "__main__":
	run(args.epochs, f"iter{args.iternum}")