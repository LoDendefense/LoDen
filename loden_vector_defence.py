from models import *
from constants import *
import pandas as pd
import numpy as np
import copy, os, random
import matplotlib.pyplot as plt
# from plot_grads import *
# from torch.utils.tensorboard import SummaryWriter
from common import DEVICE
torch.set_printoptions(precision=4, sci_mode=False)
import sys

f = open("honest_stats_{}_defence.txt","a")
sys.stdout = f
sys.stderr = f


class Organizer():
    def __init__(self, train_epoch=TRAIN_EPOCH):
        self.set_random_seed()
        self.reader = DataReader()
        self.target = TargetModel(self.reader)
        self.bar_recorder = 0
        self.last_acc = 0
        self.attack_round = 0

    def exit_bar(self, acc, threshold, bar):
        if acc - self.last_acc <= threshold:
            self.bar_recorder += 1
        else:
            self.bar_recorder = 0
        self.last_acc = acc
        return self.bar_recorder > bar

    def print_log(self,participant,participant_hist,participant_nonmember,attack_ground,nonmember_ground):
        logger.info("This is participant {}".format(participant))
        # print(participant_hist,participant_nonmember)
        final_part = {}
        final_nonmember_part = {}
        for i in participant_hist :
            # sample_list = []
            for key, value in i.items() :
                # print(value)
                if key in final_part.keys() :
                    # print(final_part1)
                    sample_list = final_part[key]
                    sample_list.append(value)
                    final_part[key] = sample_list
                else :
                    sample_list = []
                    sample_list.append(value)
                    final_part[key] = sample_list

        for i in participant_nonmember :
            # nonmember_list = []
            for key, value in i.items() :
                # print(value)
                if key in final_nonmember_part.keys() :
                    # print(final_part1)
                    nonmember_list = final_nonmember_part[key]
                    nonmember_list.append(value)
                    final_part[key] = nonmember_list
                else :
                    nonmember_list = []
                    nonmember_list.append(value)
                    final_nonmember_part[key] = nonmember_list
                    # print(final_part1)
        i = 0
        logger.info("attacker sample labels !!!")
        # print(final_nonmember_part1)
        # print(final_part1)

        for key, value in final_part.items() :
            if i >= len(attack_ground) :
                break
            else :
                # print(attack_ground[i][0],attack_ground[i][1])
                logger.info("sample index {},predicted label {}".format(key, value))
                logger.info("sample index {}, ground truth {}".format(attack_ground[i][0], attack_ground[i][1]))
                i += 1
                # print(i)
        i = 0
        logger.info("nonmember sample labels !!!")
        for key, value in final_nonmember_part.items() :
            # print(nonmember_ground[i][0], nonmember_ground[i][1])
            logger.info("sample index {},predicted label {}".format(key, value))
            logger.info("sample index {}, ground truth {}".format(nonmember_ground[i][0], nonmember_ground[i][1]))
            i += 1

    def set_random_seed(self, seed=GLOBAL_SEED):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def honest_test(self, logger, adaptive=False, record_process=True, record_model=False, plot=False):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
                                                "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                                                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        attacker_success_round = []
        # predicted_vector_collector = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(), DEFAULT_AGR)
        logger.info("AGR is {}".format(DEFAULT_AGR))
        logger.info("Dataset is {}".format(DEFAULT_SET))
        logger.info("Member ratio is {}".format(BLACK_BOX_MEMBER_RATE))
        member_total_samples = self.reader.get_honest_node_member()
        all_member_samples = self.reader.get_train_set(0)
        print(all_member_samples)
        last_batch = self.reader.get_last_train_batch()
        print(last_batch)
        number_of_round = len(member_total_samples) // 5
        member_total_samples = member_total_samples[0:100]
        print(member_total_samples)
        print(member_total_samples)
        # Initialize attacker
        logger.info(len(member_total_samples)//2)
        for l in range(len(member_total_samples)//2):
            # Initialize global model
            self.attack_round = 0
            global_model = FederatedModel(self.reader, aggregator)
            global_model.init_global_model()
            test_loss, test_acc = global_model.test_outcome()
            # Recording and printing
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (0, "g", test_loss, test_acc, 0)
            logger.info("Global model initiated, loss={}, acc={}".format(test_loss, test_acc))
            # Initialize participants
            participants = []
            for i in range(NUMBER_OF_PARTICIPANTS):
                participants.append(FederatedModel(self.reader, aggregator))
                participants[i].init_participant(global_model, i)
                # if i == 1:
                # print(participants[i].get_epoch_gradient(apply_gradient=None))
                test_loss, test_acc = participants[i].test_outcome()
                try:
                    if DEFAULT_AGR == FANG or FL_TRUST:
                        aggregator.agr_model_acquire(global_model)
                except NameError:
                    pass
                # # Recording and printing
                # if record_process:
                #     acc_recorder.loc[len(acc_recorder)] = (0, i, test_loss, test_acc, 0)
                logger.info("Participant {} initiated, loss={}, acc={}".format(i, test_loss, test_acc))

            attacker_success_round = []
            print(l)
            attacker = WhiteBoxMalicious(self.reader, aggregator)
            print(member_total_samples[0:2])
            attacker.get_samples(member_total_samples[2*l:2*l+2])
            logger.info(member_total_samples[2*l:2*l+2])
            attacker_sample_list,member_list,nonmember_list = attacker.attacker_sample()
            # member_list = member_list[:2]
            logger.info(member_list)
            for i in range(30):
                exec("honest_node_side_list%s=[]" % i)
            vector_hist = []
            honest_node_side_list=[]
            participant_0_hist = []
            participant_0_hist_nonmember = []
            grad_data = []
            loss_set = []
            correct_set = []
            removed_samples = []

            participant_1_hist =  []
            participant_1_hist_nonmember = []
            participant_2_hist = []
            participant_2_hist_nonmember = []
            participant_3_hist = []
            participant_3_hist_nonmember = []
            participant_4_hist = []
            participant_4_hist_nonmember = []
            monitor_recorder = {}
            ascent_factor = ASCENT_FACTOR
            for i in range(30):
                exec("correct_set%s = {}" % i)
            for i in range(30):
                exec("correct_set_dic%s = {}" % i)
            correct_set = []
            for j in range(MAX_EPOCH):
                # data = []
                # The global model's parameter is shared to each participant before every communication round
                global_parameters = global_model.get_flatten_parameters()
                train_acc_collector = []
                for i in range(NUMBER_OF_PARTICIPANTS):
                    # The participants collect the global parameters before training
                    participants[i].collect_parameters(global_parameters)
                    if i == 0:
                        current_hist, attack_ground_0, label_status, out_list = participants[i].check_member_label(member_list)
                        participant_0_hist.append(current_hist)
                        vector_hist.append(out_list)
                        for index in range(len(out_list)):
                            logger.info("current epoch {}, attack sample {} prediction is {} groudtruth is {}".format(j,
                                                                                                                      out_list[
                                                                                                                          index][
                                                                                                                          3],
                                                                                                                      out_list[
                                                                                                                          index][
                                                                                                                          1],
                                                                                                                      out_list[
                                                                                                                          index][
                                                                                               2]))
                        # current_nonmember_hist, nonmember_ground_0 = participants[i].check_nonmember_sample(i,
                        #                                                                                     attacker_sample_list)
<<<<<<< HEAD
                        honest_node_side,correct_set, correct_set_dic= participants[i].detect_node_side_vector(all_member_samples,last_batch)
=======
                        honest_node_side,correct_set = participants[i].detect_node_side(all_member_samples,last_batch,correct_set)
>>>>>>> parent of f392db6 (fix bugs)
                    else:
                        honest_node_side, correct_set, correct_set_dic= participants[i].nomarl_detection_vector(i)
                    # print(honest_node_side)
                    # exec("correct_set%s = correct_set" % i)
                    exec("honest_node_side_list%s = honest_node_side" % i)
                    # participant_0_hist_nonmember.append(current_nonmember_hist)
                    honest_node_side_list.append(honest_node_side)
                    # print(eval("correct_set%s" % i))
                    print(honest_node_side)
                    # print(len(honest_node_side))
                    # print(out_list)
                    # predicted_vector_collector.loc[len(predicted_vector_collector)] = (out_list)

                    # print(eval("correct_set%s" % i))
                    # print(eval("correct_set%s" % i)[0])
                    remove_counter = 0
                    if j == 100:
                        exec("correct_set%s = correct_set" % i)
                        print(eval("correct_set%s " % i))
<<<<<<< HEAD
                        exec("correct_set_dic%s = correct_set_dic" % i)
                        print(eval("correct_set_dic%s " % i))
                    if 140>j>100:
                        monitor_list = participants[i].detect_attack_vector(eval("correct_set%s " % i),eval("correct_set_dic%s" % i))
=======
                    if j>100:
                        monitor_list = participants[i].detect_attack_vector(i,eval("correct_set%s " % i))
>>>>>>> parent of f392db6 (fix bugs)
                        print(monitor_list)
                        print(eval("correct_set%s " % i))
                        if monitor_list != []:
                            for n in monitor_list:
                                # print(type(int(n.item())))
                                print(n)
<<<<<<< HEAD
                                # if n[3] in eval("correct_set%s " % i):
=======
                                # if int(n.item()) in eval("correct_set%s " % i):
>>>>>>> parent of f392db6 (fix bugs)
                            #     if n in monitor_recorder.keys():
                            #         monitor_recorder[n] = monitor_recorder.get(n)+1
                            #     else:
                            #         monitor_recorder[n] = 1
                            # logger.info(monitor_recorder)
                            # for key, velue in monitor_recorder.items():
                            #     if velue == 1:
                            #         position = key[:2]
                            #         print(key)
                            #     print(n)
                                removed_samples.append([j,i,int(n.item())])
                                # print(n)
                                participants[i].del_defence(n)
                                remove_counter+=1
                                logger.info("remove sample {}, total remove {}".format(n.item(),remove_counter))
                    logger.info("remove sample in this round {}".format(remove_counter))
                    # The participants calculate local gradients and share to the aggregator
                    participants[i].share_gradient()

                    train_loss, train_acc = participants[i].train_loss, participants[i].train_acc
                    train_acc_collector.append(train_acc)
                    # Printing and recording
                    test_loss, test_acc = participants[i].test_outcome()
                    try:
                        if DEFAULT_AGR == FANG or FL_TRUST:
                            aggregator.agr_model_acquire(global_model)
                    except NameError:
                        pass

                    logger.info(
                        "Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, i,
                                                                                                                 test_loss,
                                                                                                                 test_acc,
                                                                                                                 train_loss,
                                                                                                                 train_acc))
                # attacker attack
                attacker.collect_parameters(global_parameters)
                true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result()

                if true_member and false_member and true_non_member and false_non_member != 0:
                    attack_precision = true_member / (true_member + false_member)
                    attack_accuracy = (true_member + true_non_member) / (
                            true_member + true_non_member + false_member + false_non_member)
                    attack_recall = true_member / (true_member + false_non_member)
                else:
                    attack_precision = (true_member + 1) / (true_member + false_member + 1)
                    attack_accuracy = (true_member + true_non_member + 1) / (
                            true_member + true_non_member + false_member + false_non_member + 1)
                    attack_recall = (true_member + 1) / (true_member + false_non_member + 1)




                if j < TRAIN_EPOCH or self.attack_round > 100:
                    attacker.train()

                else:
                    logger.info("attack!!")
                    logger.info("attack {}".format(self.attack_round))
                    try:
                        attacker.optimized_gradient_ascent(ascent_factor=ascent_factor,cover_factor=COVER_FACTOR)
                    except NameError:
                        attacker.optimized_gradient_ascent(ascent_factor=ascent_factor)


                    self.attack_round += 1



                if DEFAULT_AGR in [FANG, MULTI_KRUM, KRUM]:
                    logger.info("Selected inputs are from participants number{}".format(aggregator.robust.appearence_list))
                    if 30 in aggregator.robust.appearence_list:
                        attacker_success_round.append(j)

                    logger.info("current status {}".format(str(aggregator.robust.status_list)))

                logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision,
                                                                                     attack_recall))
                pred_acc_member = attacker.evaluate_member_accuracy()[0].cpu()
                member_prediction = attacker.evaluate_member_accuracy()[1]
                logger.info("member prediction {}".format(member_prediction))
                pred_acc_non_member = attacker.evaluate_non_member_accuracy().cpu()

                attack_recorder.loc[len(attack_recorder)] = (j + 1, \
                                                             attack_accuracy, attack_precision, attack_recall, \
                                                             pred_acc_member, pred_acc_non_member, \
                                                             true_member, false_member, true_non_member, false_non_member)
                # Global model collects the aggregated gradient
                global_model.apply_gradient()
                loss_set.append(aggregator.robust.status_list+[j])
                # Printing and recording
                test_loss, test_acc = global_model.test_outcome()
                train_acc = torch.mean(torch.tensor(train_acc_collector)).item()
                self.last_acc = train_acc
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
                logger.info(
                    "Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(j + 1, test_loss, test_acc,
                                                                                            train_acc))

            # self.print_log(0,participant_0_hist,participant_0_hist_nonmember,attack_ground_0, nonmember_ground_0)

            logger.info("attack success round {}, total {}".format(attacker_success_round, len(attacker_success_round)))

            honest_node_predicted_vector_collector = pd.DataFrame(removed_samples)
            honest_node_predicted_vector_collector.to_csv(
                EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "predicted_vector" + "_round" + str(l) + str(
                    MAX_EPOCH - TRAIN_EPOCH) + "optimized_model_single_honest_side_defence_removed.csv")
            for i in range(30):
                honest_node_predicted_vector_collector = pd.DataFrame(eval("honest_node_side_list%s"%i))
                honest_node_predicted_vector_collector.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                                        "TrainEpoch" + str(TRAIN_EPOCH) + "predicted_vector" + "_round" +str(l)+str(
                        MAX_EPOCH - TRAIN_EPOCH) + "optimized_model_single_honest_side_defence_%s.csv"%i)

            predicted_vector_collector = pd.DataFrame(vector_hist)
            predicted_vector_collector.to_csv(
                EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "predicted_vector" + "_round" + str(l) + str(
<<<<<<< HEAD
                    MAX_EPOCH - TRAIN_EPOCH) + "optimized_model_single_defence_vector.csv")
=======
                    MAX_EPOCH - TRAIN_EPOCH) + "optimized_model_single_defence.csv")
>>>>>>> parent of f392db6 (fix bugs)

            if record_model:
                param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
                for i in range(NUMBER_OF_PARTICIPANTS):
                    param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
                param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + "Federated_Models_honest_side_defence.csv")
            if record_process:
                recorder_suffix = "honest_node"
                acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                                    "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                    MAX_EPOCH - TRAIN_EPOCH) + recorder_suffix + "_round" +str(l)+"optimized_model_single_ascent_factor{}_honest_side_defence.csv".format(ASCENT_FACTOR))
                attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                                       "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                    MAX_EPOCH - TRAIN_EPOCH) + recorder_suffix + "_round" +str(l) +"optimized_attacker_single_ascent_factor{}_honest_side_defence.csv".format(ASCENT_FACTOR))
            if plot:
                self.plot_attack_performance(attack_recorder)





logger = make_logger("project", EXPERIMENTAL_DATA_DIRECTORY,
                     'log_{}_{}_{}_TrainEpoch{}_AttackEpoch{}_honest_member_rate{}_ascent_factor{}_single_honest_side_defence'.format(TIME_STAMP, DEFAULT_SET, DEFAULT_AGR,
                                                                               TRAIN_EPOCH,
                                                                               MAX_EPOCH - TRAIN_EPOCH,BLACK_BOX_MEMBER_RATE,ASCENT_FACTOR))

org = Organizer()
org.set_random_seed()
org.honest_test(logger)