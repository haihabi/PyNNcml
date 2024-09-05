from torch.utils.data import Dataset


class LinkDataset(Dataset):
    def __init__(self, link_set, point_set, transform=None, target_transform=None):
        """
        Dataset of links
        :param link_set: LinkSet
        :param transform: Transform
        :param target_transform: Transform
        """
        self.link_set = link_set
        self.point_set = point_set
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Get the number of links
        """
        return self.link_set.n_links

    def __getitem__(self, idx):
        """
        Get the item of the link
        :param idx: index of the link

        :return: rain, rsl, tsl, metadata
        """
        rain, rsl, tsl, metadata = self.link_set.get_link(idx).data_alignment()
        if self.transform:
            raise NotImplemented
        if self.target_transform:
            raise NotImplemented
        return rain, rsl, tsl, metadata


class SubSequentLinkDataset(Dataset):
    def __init__(self, data, label, meta_data, transform=None, target_transform=None):
        self.data = data
        self.label = label
        self.meta_data = meta_data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rain, data, metadata = self.label[idx], self.data[idx], self.meta_data[idx]
        if self.transform:
            raise NotImplemented
        if self.target_transform:
            raise NotImplemented

        return rain, data, metadata
