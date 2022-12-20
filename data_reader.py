import pandas as pd
import numpy as np
import torch
import torchvision as tv
from torch.utils.data import DataLoader
from constants import *
from common import DEVICE
import torch.utils.data as Data

# class CelebA(torch.utils.data.Dataset):
#     base_folder = "celeba"
#
#     def __init__(
#             self,
#             root: str,
#             attr_list: str,
#             target_type: Union[List[str], str] = "attr",
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#     ) -> None:
#
#         if isinstance(target_type, list):
#             self.target_type = target_type
#         else:
#             self.target_type = [target_type]
#
#         self.root = root
#         self.transform = transform
#         self.target_transform =target_transform
#         self.attr_list = attr_list
#
#         fn = partial(os.path.join, self.root, self.base_folder)
#         splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
#         attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
#
#         mask = slice(None)
#
#         self.filename = splits[mask].index.values
#         self.attr = torch.as_tensor(attr[mask].values)
#         self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
#         self.attr_names = list(attr.columns)
#
#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         X = Image.open(os.path.join(self.root, self.base_folder, "img_celeba", self.filename[index]))
#
#         target: Any = []
#         for t, nums in zip(self.target_type, self.attr_list):
#             if t == "attr":
#                 final_attr = 0
#                 for i in range(len(nums)):
#                     final_attr += 2 ** i * self.attr[index][nums[i]]
#                 target.append(final_attr)
#             else:
#                 # TODO: refactor with utils.verify_str_arg
#                 raise ValueError("Target type \"{}\" is not recognized.".format(t))
#
#         if self.transform is not None:
#             X = self.transform(X)
#
#         if target:
#             target = tuple(target) if len(target) > 1 else target[0]
#
#             if self.target_transform is not None:
#                 target = self.target_transform(target)
#         else:
#             target = None
#
#         return X, target
#
#     def __len__(self) -> int:
#         return len(self.attr)
#
#     def extra_repr(self) -> str:
#         lines = ["Target type: {target_type}", "Split: {split}"]
#         return '\n'.join(lines).format(**self.__dict__)


class DataReader:
    """
    The class to read data set from the given file
    """
    def __init__(self, data_set=DEFAULT_SET, label_column=LABEL_COL, batch_size=BATCH_SIZE,
                 distribution=DEFAULT_DISTRIBUTION, reserved=0):
        """
        Load the data from the given data path
        :param path: the path of csv file to load data
        :param label_column: the column index of csv file to store the labels
        :param label_size: The number of overall classes in the given data set
        """
        # load the csv file
        if data_set == PURCHASE100:
            path = PURCHASE100_PATH
            data_frame = pd.read_csv(path, header=None)
            # extract the label
            self.labels = torch.tensor(data_frame[label_column].to_numpy(), dtype=torch.int64).to(DEVICE)
            self.labels -= 1
            data_frame.drop(label_column, inplace=True, axis=1)
            # extract the data
            self.data = torch.tensor(data_frame.to_numpy(), dtype=torch.float).to(DEVICE)



        # elif data_set == CIFAR100:
        #     # Normalize input
        #     CIFAR100_transform = tv.transforms.Compose([
        #             tv.transforms.Pad(4),
        #             tv.transforms.RandomCrop(32),
        #             tv.transforms.RandomHorizontalFlip(),
        #             tv.transforms.ToTensor(),
        #             tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #         ])
        #
        #     cifar100 = tv.datasets.CIFAR100(CIFAR100_PATH, transform=CIFAR100_transform, download=False)
        #     loader = DataLoader(cifar100, batch_size=len(cifar100))
        #     self.data = next(iter(loader))[0]
        #     self.data = torch.flatten(self.data, 1)
        #     self.labels = next(iter(loader))[1]
        #     self.data = self.data.float()
        #     self.labels = self.labels.long()

        elif data_set == CIFAR_10:
            samples = np.vstack(
                [np.genfromtxt(CIFAR_10_PATH+"train{}.csv".format(x), delimiter=',') for x in range(4)]
            )
            self.data = torch.tensor(samples[:, :-1], dtype=torch.float).to(DEVICE)
            self.labels = torch.tensor(samples[:, -1], dtype=torch.int64).to(DEVICE)

        elif data_set == LOCATION30:
            path = LOCATION30_PATH
            data_frame = pd.read_csv(path, header=None)
            # extract the label
            self.labels = torch.tensor(data_frame[label_column].to_numpy(), dtype=torch.int64).to(DEVICE)
            self.labels -= 1
            data_frame.drop(label_column, inplace=True, axis=1)
            # extract the data
            self.data = torch.tensor(data_frame.to_numpy(), dtype=torch.float).to(DEVICE)
        elif data_set == TEXAS100:
            path = TEXAS100_PATH
            self.data = np.load(path)
            self.labels = self.data['labels']
            self.data = self.data['features']
            self.labels = np.argmax(self.labels, axis=1)
            self.labels = torch.tensor(self.labels, dtype=torch.int64).to(DEVICE)
            self.data = torch.tensor(self.data, dtype=torch.float).to(DEVICE)
        elif data_set == MNIST:
            # Normalize input
            MNIST_transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=[0.5, ],
                    std=[0.5, ])
            ])
            mnist = tv.datasets.MNIST(MNIST_PATH, transform=MNIST_transform, download=True)
            loader = DataLoader(mnist, batch_size=len(mnist))
            self.data = next(iter(loader))[0]
            self.data = torch.flatten(self.data, 1)
            self.labels = next(iter(loader))[1]
            self.data = self.data.float()
            self.labels = self.labels.long()
        elif data_set == GNOME:
            loaded = np.load(GNOME_PATH)
            self.data = torch.tensor(loaded['features'], dtype=torch.float).to(DEVICE)
            self.labels = torch.tensor(loaded['labels'], dtype=torch.int64).to(DEVICE)

        self.data = self.data.to(DEVICE)
        self.labels = self.labels.to(DEVICE)
        self.empty_data = torch.zeros_like(self.data[0]).to(DEVICE)
        self.empty_label = torch.tensor(200,dtype=torch.int64)
        self.empty_indice = torch.tensor([-1]).to(DEVICE)
        self.skip = False
        self.train_set_0 = None

        try:
            reserved = RESERVED_SAMPLE
        except NameError:
            reserved = 0

        try:
            fl_trust_samples = FL_TRUST_SET
        except NameError:
            fl_trust_samples = 0

        # initialize the training and testing batches indices
        self.train_set = None
        self.test_set = None
        self.train_set_last_batch = None
        self.test_set_last_batch =None
        overall_size = self.labels.size(0)
        # overall_size = 12000
        if distribution is None:
            # divide data samples into batches, drop the last bit of data samples to make sure each batch is full sized
            overall_size -= 16
            rand_perm = torch.randperm(self.labels.size(0)).to(DEVICE)
            self.reserve_set = rand_perm[overall_size:]
            print("cover dataset size is {}".format(reserved))
            overall_size -= fl_trust_samples
            self.fl_trust = rand_perm[overall_size+reserved:]
            print(len(self.fl_trust))
            print("FL TRUST dataset size is {}".format(fl_trust_samples))
            print(overall_size) #12000
            print(overall_size-overall_size % batch_size) #11968
            all_size = overall_size
            overall_size -= overall_size % batch_size
            rand_perm_last = rand_perm[overall_size:all_size]
            rand_perm = rand_perm[:overall_size]
            print(rand_perm_last)
            # self.last_batch_indices = rand_perm_last.reshape((-1, all_size-overall_size)).to(DEVICE)
            # print(self.last_batch_indices)
            # print(len(self.last_batch_indices))
            self.last_batch_indices = self.reserve_set
            self.batch_indices = rand_perm.reshape((-1, batch_size)).to(DEVICE)
            # print(self.batch_indices)
            self.train_test_split()
            # dataset = Data.TensorDataset(self.data,self.labels)
            # loader = Data.DataLoader(dataset = dataset, batch_size= BATCH_SIZE, shuffle=True)
            # print(loader)
        elif distribution == CLASS_BASED:
            self.train_test_split(batch_training=False)
            print("data split, train set length={}, test set length={}".format(len(self.train_set), len(self.test_set)))
            self.class_indices = {}
            self.class_training = {}
            self.class_testing = {}
            for i in range(torch.max(self.labels).item()+1):
                self.class_indices[i] = torch.nonzero((self.labels == i)).to(DEVICE)
                self.class_training[i] = torch.tensor(np.intersect1d(self.class_indices[i], self.train_set)).to(DEVICE)
                self.class_testing[i] = torch.tensor(np.intersect1d(self.class_indices[i], self.test_set)).to(DEVICE)
                print("Label {}, overall samples ={}, train_set={}, test_set={}".format(i, len(self.class_indices[i]), len(self.class_training[i]), len(self.class_testing[i])))

        # print(self.batch_indices.size())

        print("Data set "+DEFAULT_SET+
              " has been loaded, overall {} records, batch size = {}, testing batches: {}, training batches: {}"
              .format(overall_size, batch_size, self.test_set.size(0), self.train_set.size(0)))

    def get_fl_trust_set(self):
        return self.fl_trust

    def train_test_split(self, ratio=TRAIN_TEST_RATIO, batch_training=BATCH_TRAINING):
        """
        Split the data set into training set and test set according to the given ratio
        :param ratio: tuple (float, float) the ratio of train set and test set
        :param batch_training: True to train by batch, False will not
        :return: None
        """
        if batch_training:
            train_count = round(self.batch_indices.size(0) * ratio[0] / sum(ratio))
            last_count = 8
            # print(last_count)
            self.train_set = self.batch_indices[:train_count].to(DEVICE)
            self.test_set = self.batch_indices[train_count:].to(DEVICE)
            self.train_set_last_batch = self.last_batch_indices[:last_count].to(DEVICE)
            # self.train_set_last_batch = torch.tensor([]).to(DEVICE)
            self.test_set_last_batch = self.last_batch_indices[last_count:].to(DEVICE)
            # print(self.train_set_last_batch)
        else:
            train_count = round(self.data.size(0) * ratio[0] / sum(ratio))
            rand_perm = torch.randperm(self.data.size(0)).to(DEVICE)
            self.train_set = rand_perm[:train_count].to(DEVICE)
            self.test_set = rand_perm[train_count:].to(DEVICE)

    def refresh_batches(self,participant_index, remaining_batches, last_batch):
        remaining_batches =torch.flatten(remaining_batches)
        last_batch = torch.flatten(last_batch)
        remain_indices = torch.cat((remaining_batches,last_batch),0)
        last_batch_size = remain_indices.size(0)//NUMBER_OF_PARTICIPANTS
        lower_bound = participant_index * last_batch_size
        upper_bound = (participant_index + 1) * last_batch_size
        return remain_indices[lower_bound: upper_bound]


    def get_train_set(self, participant_index=0, distribution=DEFAULT_DISTRIBUTION, by_batch=BATCH_TRAINING, batch_size=BATCH_SIZE):
        """
        Get the indices for each training batch
        :param participant_index: the index of a particular participant, must be less than the number of participants
        :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each training batch
        """
        if distribution is None:
            # if participant_index == 0:
            #     return self.train_set[: 3]
            # else:
            batches_per_participant = self.train_set.size(0) // NUMBER_OF_PARTICIPANTS
            remain_batch = self.train_set.size(0) - batches_per_participant * NUMBER_OF_PARTICIPANTS
            remaining_batch = self.train_set[NUMBER_OF_PARTICIPANTS * batches_per_participant:]
            # self.train_set_last_batch = self.refresh_batches(participant_index,remaining_batch,self.train_set_last_batch)
            lower_bound = participant_index * batches_per_participant
            upper_bound = (participant_index + 1) * batches_per_participant
            # self.train_set_0 = self.train_set[lower_bound: upper_bound].to(DEVICE)
            # if participant_index == 0 and self.train_set_0!=None:
            #     return self.train_set_0
            # elif participant_index == 0 and self.train_set_0 == None:
            #     self.train_set_0 = self.train_set[lower_bound: upper_bound].to(DEVICE)
            #     return self.train_set[lower_bound: upper_bound]
            # else:
                # self.train_set_0 = self.train_set[lower_bound: upper_bound].to(DEVICE)
            return self.train_set[lower_bound: upper_bound]
        if distribution == CLASS_BASED:
            class_count = torch.max(self.labels).item() + 1
            class_per_participant = class_count // NUMBER_OF_PARTICIPANTS
            my_set = []
            lower_bound = participant_index * class_per_participant
            upper_bound = (participant_index + 1) * class_per_participant
            for i in range(lower_bound, upper_bound):
                my_set.append(self.class_training[i])
            if participant_index == NUMBER_OF_PARTICIPANTS - 1:
                for i in range(upper_bound, class_count):
                    my_set.append(self.class_training[i])
            all_samples = torch.hstack(my_set)
            if by_batch:
                lenth = len(all_samples)
                lenth -= lenth % batch_size
                all_samples = all_samples[:lenth].reshape((-1, batch_size))
            # print("The size of training set for participant {} is {}".format(participant_index, all_samples.size()))
            return all_samples

    def get_test_set(self, participant_index=0, distribution=DEFAULT_DISTRIBUTION, by_batch=BATCH_TRAINING, batch_size=BATCH_SIZE):
        """
        Get the indices for each test batch
        :param participant_index: the index of a particular participant, must be less than the number of participants
        :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each test batch
        """
        if distribution is None:
            # if participant_index == 0:
            #     return self.test_set[: 3]
            # else:
            batches_per_participant = self.test_set.size(0) // NUMBER_OF_PARTICIPANTS
            remaining_batch = self.test_set[NUMBER_OF_PARTICIPANTS * batches_per_participant:]
            # self.test_set_last_batch = self.refresh_batches(participant_index, remaining_batch,
            #                                                  self.test_set_last_batch)
            print(self.test_set_last_batch)
            lower_bound = participant_index * batches_per_participant
            upper_bound = (participant_index + 1) * batches_per_participant
            return self.test_set[lower_bound: upper_bound]
        elif distribution == CLASS_BASED:
            class_count = torch.max(self.labels).item() + 1
            class_per_participant = class_count // NUMBER_OF_PARTICIPANTS
            my_set = []
            lower_bound = participant_index * class_per_participant
            upper_bound = (participant_index + 1) * class_per_participant
            for i in range(lower_bound, upper_bound):
                my_set.append(self.class_testing[i])
            if participant_index == NUMBER_OF_PARTICIPANTS - 1:
                for i in range(upper_bound, class_count):
                    my_set.append(self.class_testing[i])
            all_samples = torch.hstack(my_set)
            if by_batch:
                lenth = len(all_samples)
                lenth -= lenth % batch_size
                all_samples = all_samples[:lenth].reshape((-1, batch_size))
            # print("The size of testing set for participant {} is {}".format(participant_index, all_samples.size()))
            return all_samples

    def get_last_train_batch(self):
        # self.train_set_last_batch = self.train_set_last_batch
        return self.train_set_last_batch

    def get_last_test_batch(self):
        self.test_set_last_batch = self.test_set_last_batch.reshape((-1))
        return self.test_set_last_batch

    def get_batch(self, batch_indices):
        """
        Get the batch of data according to given batch indices
        :param batch_indices: tensor[BATCH_SIZE], the indices of a particular batch
        :return: tuple (tensor, tensor) the tensor representing the data and labels
        """
        return self.data[batch_indices], self.labels[batch_indices]

    def get_honest_node_member(self,participant_index = 0):
        member_list = []
        train_flatten = self.train_set.flatten().to(DEVICE)
        # test_flatten = self.test_set.flatten().to(DEVICE)
        for j in range(len(train_flatten)):
            if train_flatten[j] in self.get_train_set(participant_index):
                # if len(member_list) < participant_member_count:
                member_eachx, member_eachy = self.get_batch(train_flatten[j])
                print(member_eachy)
                member_list.append(train_flatten[j])
        member_total = torch.tensor(member_list).to(DEVICE)
        return member_total

    def get_honest_node_nonmember(self,participant_index = 0):
        self.train_set = torch.concat(self.train_set,self.reserve_set)



    def get_black_box_batch(self, member_rate=BLACK_BOX_MEMBER_RATE, attack_batch_size=NUMBER_OF_ATTACK_SAMPLES):
        """
        Generate batches for black box training
        :param member_rate The rate of member data samples
        :param attack_batch_size the number of data samples allocated to the black-box attacker
        """
        member_count = round(attack_batch_size * member_rate)
        non_member_count = attack_batch_size - member_count
        train_flatten = self.train_set.flatten().to(DEVICE)
        test_flatten = self.test_set.flatten().to(DEVICE)
        member_indices = train_flatten[torch.randperm(len(train_flatten))[:member_count]].to(DEVICE)
        non_member_indices = test_flatten[2:][torch.randperm((len(test_flatten)))[:non_member_count]].to(DEVICE)
        result = torch.cat([member_indices, non_member_indices]).to(DEVICE)
        result = result[torch.randperm(len(result))].to(DEVICE)
        return result, member_indices, non_member_indices


    def get_black_box_batch_fixed(self, member_rate=BLACK_BOX_MEMBER_RATE, attack_batch_size=NUMBER_OF_ATTACK_SAMPLES):
        """
        Generate batches for black box training
        :param member_rate The rate of member data samples
        :param attack_batch_size the number of data samples allocated to the black-box attacker
        """
        # member_count = 2
        participant_member_count = 2
        member_list = []
        non_member_count = attack_batch_size - participant_member_count
        participant_nonmember_count = non_member_count // NUMBER_OF_PARTICIPANTS
        train_flatten = self.train_set.flatten().to(DEVICE)
        test_flatten = self.test_set.flatten().to(DEVICE)
        for j in range(len(train_flatten)):
            if train_flatten[j] in self.get_train_set(0):
                if len(member_list) < participant_member_count:
                    member_eachx, member_eachy = self.get_batch(train_flatten[j])
                    print(member_eachy)
                    if int(member_eachy) == 2:
                        member_list.append(train_flatten[j])

                else:
                    break
        # member_indices = self.train_set[0][:participant_member_count].to(DEVICE)
        member_indices = torch.tensor(member_list).to(DEVICE)
        # print(self.train_set[0])
        print(member_indices)
        member_class_list = []
        # non_member_indices = self.train_set[0][:participant_nonmember_count].to(DEVICE)
        # member_indices = torch.cat((member_indices,self.train_set[0][:participant_member_count]),0).to(DEVICE)
        print(member_indices)
        member_x,member_y = self.get_batch(member_indices)
        for i in member_y:
            # print(i)
            member_class_list.append(i)
        # print(member_class_list)
            # non_member_indices = torch.cat((non_member_indices, self.train_set[i + 1][:participant_nonmember_count]),0).to(DEVICE)
        # print(len(member_indices),member_indices)
        # member_indices = train_flatten[range(len(train_flatten))[:member_count]].to(DEVICE)
        # non_member_indices = train_flatten[range(len(train_flatten))[:non_member_count]].to(DEVICE)
        # member_indices = train_flatten[torch.randperm(len(train_flatten))[:member_count]].to(DEVICE)
        same_class_list = []
        # print(len(test_flatten))
        diff = 0
        for index,i in enumerate(test_flatten):
            test_x, test_y = self.get_batch(i)
            if test_y not in member_class_list:
                same_class_list.append(i)
        diff_class_test_flatten = torch.tensor(same_class_list)
        # print(same_class_list)
        # for i in same_class_list:
        #     test_flatten = self.del_samples(i,test_flatten)
        # print(len(diff_class_test_flatten))
        non_member_indices = diff_class_test_flatten[torch.randperm((len(diff_class_test_flatten)))[:non_member_count]].to(DEVICE)
        non_member_x,nonmember_y = self.get_batch(non_member_indices)
        # print(2 in nonmember_y)
        # print(len(member_indices),len(non_member_indices))
        result = torch.cat([member_indices, non_member_indices]).to(DEVICE)
        # result = result[torch.randperm(len(result))].to(DEVICE)
        print(len(result))
        print(member_indices)
        return result, member_indices, non_member_indices

    def get_black_box_batch_fixed_single(self, member_rate=BLACK_BOX_MEMBER_RATE, attack_batch_size=NUMBER_OF_ATTACK_SAMPLES):
        """
        Generate batches for black box training
        :param member_rate The rate of member data samples
        :param attack_batch_size the number of data samples allocated to the black-box attacker
        """
        # member_count = 2
        participant_member_count = 2
        member_list = []
        non_member_count = attack_batch_size - participant_member_count
        participant_nonmember_count = non_member_count // NUMBER_OF_PARTICIPANTS
        train_flatten = self.train_set.flatten().to(DEVICE)
        test_flatten = self.test_set.flatten().to(DEVICE)
        for i in self.get_train_set(0):
            if len(member_list) < participant_member_count:
                member_eachx, member_eachy = self.get_batch(i)
                if member_eachy == 2:
                    member_list.append(i)
            else:
                break
        # member_indices = self.train_set[0][:participant_member_count].to(DEVICE)
        member_indices = torch.tensor(member_list).to(DEVICE)
        # print(self.train_set[0])
        # print(member_indices)
        member_class_list = []
        # non_member_indices = self.train_set[0][:participant_nonmember_count].to(DEVICE)
        # member_indices = torch.cat((member_indices,self.train_set[0][:participant_member_count]),0).to(DEVICE)
        member_x,member_y = self.get_batch(member_indices)
        for i in member_y:
            # print(i)
            member_class_list.append(i)
        # print(member_class_list)
            # non_member_indices = torch.cat((non_member_indices, self.train_set[i + 1][:participant_nonmember_count]),0).to(DEVICE)
        # print(len(member_indices),member_indices)
        # member_indices = train_flatten[range(len(train_flatten))[:member_count]].to(DEVICE)
        # non_member_indices = train_flatten[range(len(train_flatten))[:non_member_count]].to(DEVICE)
        # member_indices = train_flatten[torch.randperm(len(train_flatten))[:member_count]].to(DEVICE)
        same_class_list = []
        # print(len(test_flatten))
        diff = 0
        for index,i in enumerate(test_flatten):
            test_x, test_y = self.get_batch(i)
            if test_y not in member_class_list:
                same_class_list.append(i)
        diff_class_test_flatten = torch.tensor(same_class_list)
        # print(same_class_list)
        # for i in same_class_list:
        #     test_flatten = self.del_samples(i,test_flatten)
        # print(len(diff_class_test_flatten))
        non_member_indices = diff_class_test_flatten[torch.randperm((len(diff_class_test_flatten)))[:non_member_count]].to(DEVICE)
        non_member_x,nonmember_y = self.get_batch(non_member_indices)
        # print(2 in nonmember_y)
        # print(len(member_indices),len(non_member_indices))
        result = torch.cat([member_indices, non_member_indices]).to(DEVICE)
        # result = result[torch.randperm(len(result))].to(DEVICE)
        print(len(result))
        print(member_indices)
        return result, member_indices, non_member_indices

    def get_black_box_batch_fixed_balance_class(self, member_rate=BLACK_BOX_MEMBER_RATE, attack_batch_size=NUMBER_OF_ATTACK_SAMPLES):
        """
        Generate batches for black box training
        :param member_rate The rate of member data samples
        :param attack_batch_size the number of data samples allocated to the black-box attacker
        """
        # member_count = 2
        participant_member_count = 2
        member_list = []
        non_member_count = attack_batch_size - participant_member_count
        participant_nonmember_count = non_member_count // NUMBER_OF_PARTICIPANTS
        train_flatten = self.train_set.flatten().to(DEVICE)
        test_flatten = self.test_set.flatten().to(DEVICE)
        for j in range(len(train_flatten)):
            if train_flatten[j] in self.get_train_set(0):
                if len(member_list) < participant_member_count:
                    member_eachx, member_eachy = self.get_batch(train_flatten[j])
                    print(member_eachy)
                    if int(member_eachy) == 2:
                        member_list.append(train_flatten[j])

                else:
                    break

        # member_indices = self.train_set[0][:participant_member_count].to(DEVICE)
        member_indices = torch.tensor(member_list).to(DEVICE)
        # print(self.train_set[0])
        # print(member_indices)
        member_class_list = []
        # non_member_indices = self.train_set[0][:participant_nonmember_count].to(DEVICE)
        # member_indices = torch.cat((member_indices,self.train_set[0][:participant_member_count]),0).to(DEVICE)
        print(member_indices)
        member_x,member_y = self.get_batch(member_indices)
        for i in member_y:
            # print(i)
            member_class_list.append(i)
        # print(member_class_list)
            # non_member_indices = torch.cat((non_member_indices, self.train_set[i + 1][:participant_nonmember_count]),0).to(DEVICE)
        # print(len(member_indices),member_indices)
        # member_indices = train_flatten[range(len(train_flatten))[:member_count]].to(DEVICE)
        # non_member_indices = train_flatten[range(len(train_flatten))[:non_member_count]].to(DEVICE)
        # member_indices = train_flatten[torch.randperm(len(train_flatten))[:member_count]].to(DEVICE)
        # same_class_list = []
        # # print(len(test_flatten))
        # diff = 0
        # for index,i in enumerate(test_flatten):
        #     test_x, test_y = self.get_batch(i)
        #     if test_y not in member_class_list:
        #         same_class_list.append(i)
        # diff_class_test_flatten = torch.tensor(same_class_list)
        # print(same_class_list)
        # for i in same_class_list:
        #     test_flatten = self.del_samples(i,test_flatten)
        # print(len(diff_class_test_flatten))
        non_member_indices = test_flatten[torch.randperm((len(test_flatten)))[:non_member_count]].to(DEVICE)
        non_member_x,nonmember_y = self.get_batch(non_member_indices)
        print(nonmember_y)
        # print(len(member_indices),len(non_member_indices))
        result = torch.cat([member_indices, non_member_indices]).to(DEVICE)
        # result = result[torch.randperm(len(result))].to(DEVICE)
        print(len(result))
        print(member_indices)
        return result, member_indices, non_member_indices

    def del_samples(self,index,train_set,last_batch):

        flatten_set = train_set.flatten()
        if last_batch!= None:
            print(flatten_set)
            print(last_batch.flatten())
            flatten_set = torch.cat((flatten_set,last_batch.flatten()),0)
        print(len(flatten_set))
        flatten_set = flatten_set.cpu().numpy().tolist()
        # remove_counter = 0
        print(index)
        # for i in index:
        flatten_set.remove(index)
        # remove_counter+=1
        # rand_perm = torch.randperm(self.labels.size(0)).to(DEVICE)
        over_train_size = len(flatten_set)
        print(over_train_size)
        over_full_size = over_train_size- over_train_size % BATCH_SIZE
        # last_batch_size = over_train_size-over_full_size
        full_indeices = torch.tensor(flatten_set[:over_full_size]).to(DEVICE)
        train_set = full_indeices.reshape((-1, BATCH_SIZE)).to(DEVICE)
        last_indeices = torch.tensor(flatten_set[over_full_size:over_train_size]).to(DEVICE)
        last_batch = last_indeices
        print(last_batch)

        return train_set,last_batch