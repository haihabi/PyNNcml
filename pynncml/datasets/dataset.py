from torch.utils.data import Dataset


class LinkDataset(Dataset):
    def __init__(self, link_set, transform=None, target_transform=None):
        self.link_set = link_set
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.link_set.n_links

    def __getitem__(self, idx):
        rain, rsl, tsl, metadata = self.link_set.get_link(idx).data_alignment()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return rain, rsl, tsl, metadata
