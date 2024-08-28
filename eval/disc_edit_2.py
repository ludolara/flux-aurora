from argparse import ArgumentParser
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from diffusers import DiffusionPipeline
from edit_dataset import EditITMDataset
from dotenv import load_dotenv
load_dotenv()

class DiscEditMetric:
    def __init__(self, model_name: str, device: str = 'cuda'):
        """
        Initialize the DiscEditMetric class with a DiffusionPipeline model and device.

        Args:
            model_name (str): The name of the pre-trained model to load.
            device (str): Device to run the model on, default is 'cuda'.
        """
        self.device = device
        self.pipe = DiffusionPipeline.from_pretrained(model_name, cache_dir=os.getenv("CACHE_DIR")).to(device)
        self.to_tensor = transforms.ToTensor()

    def encode_latents(self, images):
        """
        Encode images into the latent space using the diffusion model's encoder.

        Args:
            images (torch.Tensor): Batch of images to encode.

        Returns:
            torch.Tensor: Encoded latent representations.
        """
        images = images.to(self.device)
        # Check if the pipeline has a VAE
        if hasattr(self.pipe, "vae"):
            # Encode the images to latent space using the VAE
            with torch.no_grad():
                latents = self.pipe.vae.encode(images).latent_dist.sample()
        else:
            raise AttributeError("The pipeline does not have a VAE for encoding images.")
        return latents

    def compute_metric(self, dataset):
        """
        Compute the DiscEdit metric for the given dataset.

        Args:
            dataset (Dataset): A PyTorch Dataset containing (source_image, prompt_no_change, prompt_change) tuples.

        Returns:
            float: The DiscEdit score representing the model's discriminative ability.
        """
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        correct_predictions = 0
        total_predictions = 0

        for batch in dataloader:
            input_images = batch["input"].to(self.device)
            # texts = [pos, neg]
            prompt_change, prompt_no_change = batch["texts"]

            # Generate images
            generated_change = self.pipe(prompt_change[0], image=input_images).images[0]
            generated_no_change = self.pipe(prompt_no_change[0], image=input_images).images[0]

            # Convert generated images to tensors and move to the device
            generated_change_tensor = self.to_tensor(generated_change).to(self.device)
            generated_no_change_tensor = self.to_tensor(generated_no_change).to(self.device)

            # print(input_images.shape)
            # print(generated_change_tensor.shape)
            # print(generated_no_change_tensor.shape)

            # Encode images into the latent space
            # latent_source = self.encode_latents(input_images)
            # latent_no_change = self.encode_latents(generated_no_change)
            # latent_change = self.encode_latents(generated_change)
            with torch.no_grad():
                latent_source = self.pipe.vae.encode(input_images).latent_dist.sample()
                latent_change = self.pipe.vae.encode(generated_change_tensor.unsqueeze(0)).latent_dist.sample()
                latent_no_change = self.pipe.vae.encode(generated_no_change_tensor.unsqueeze(0)).latent_dist.sample()

                #   qq  uitar batch size!!
                # Calculate L2 distances 
                distance_change = torch.norm(latent_source - latent_change, dim=1)
                distance_no_change = torch.norm(latent_source - latent_no_change, dim=1)

                print(distance_change)
                print(distance_no_change)

                # Apply the DiscEdit metric
                disc_edit_score = 1 if distance_no_change < distance_change else 0

                print(disc_edit_score)
                # correct_predictions += (distance_no_change < distance_change).sum().item()
                # total_predictions += len(source_images)

                # print(correct_predictions)
                print("-----------------")


        return correct_predictions / 200

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="aurora-mixratio-15-15-1-1-42k-steps.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--task", default='whatsup', type=str)
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--samples", default=4, type=int)
    parser.add_argument("--size", default=512, type=int)
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument('--targets', type=str, nargs='*', help="which target groups for mmbias",default='')
    parser.add_argument("--device", default=0, type=int, help="GPU device index")
    parser.add_argument("--log_imgs", action="store_true")
    parser.add_argument("--conditional_only", action="store_true")
    parser.add_argument("--metric", default="latent", type=str)
    parser.add_argument("--split", default='valid', type=str)
    parser.add_argument("--skip", default=1, type=int)
    args = parser.parse_args()


    dataset = EditITMDataset(split=args.split, task=args.task, min_resize_res=args.size, max_resize_res=args.size, crop_res=args.size)

    model_name = "McGill-NLP/AURORA"
    disc_edit = DiscEditMetric(model_name)
    score = disc_edit.compute_metric(dataset)
    print(f"DiscEdit Score: {score:.4f}")

if __name__ == "__main__":
    main()
