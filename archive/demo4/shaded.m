clear
close all

%%

tmp = readtable("results/test_train12/4-0.csv");
tmp = table2array(tmp);
tmp = tmp(:,1);
plot(tmp)


%%
TRAIN_NAME = "test_train12";
% TRAIN_NAME = "main_result";

root_dir = "results/" + TRAIN_NAME;
root_info = dir(root_dir);
root_num = length(root_info);

%% data_ori (1000 70)
data_ori = [];   
for j = 3:root_num
    tmp = readtable(root_dir + "/" + root_info(j).name);
    tmp = table2array(tmp);
    tmp = tmp(:,1);
    data_ori = [data_ori tmp];
end

%%
rep_num = 1;
[episode_num, case_num] = size(data_ori);
case_num = case_num / rep_num;
%% expected [1000 7 10]

% data = data_ori(1,[1:10]) % for 1 episode, 1 case (episode,case)

data_std_list = zeros(case_num,episode_num); 
data_mean_list = zeros(case_num,episode_num);

for i = 1:1:case_num
    tmp = data_ori(:,(1:rep_num) + rep_num*(i-1))';
    data_std_list(i,:) = std(tmp);
    data_mean_list(i,:) = mean(tmp);
end
%%
avg_range = 5;
% color = ['r','g','k', 'c', 'm','y','b'];
color = ['r','g','k', 'b', 'm','y','c'];
data_legend = string(1:1:case_num);
% data_legend = ["eps0.9","eps0.2","0.01","0.25","0.5","0.75","0.99"];

x_tmp = 1:episode_num;
% 
% %% overall fig
% figure(1)
% for i=1:case_num
%     sample_mean = smooth(data_mean_list(i,:), avg_range)';
%     sample_std = smooth(data_std_list(i,:), avg_range)';
% 
%     hold on
%     curve1 = sample_mean - sample_std;
%     curve2 = sample_mean + sample_std;
%     inBetween = [curve1, fliplr(curve2)];
%     fill([x_tmp fliplr(x_tmp)],inBetween, color(i), 'LineStyle', 'None', 'FaceAlpha', 0.35);
% end
% 
% for i=1:case_num
%     sample_mean = smooth(data_mean_list(i,:), avg_range)';
%     sample_std = smooth(data_std_list(i,:), avg_range)';
%     
%     hold on;
%     plot(sample_mean, color(i), "LineWidth", 2);
% end
% 
% xlabel("train episode")
% ylabel("reward")
% legend(data_legend(1:end), "location", "southeast")

%% selected fig
selected = [5, 10];
figure(2)
for i=selected
    sample_mean = smooth(data_mean_list(i,:), avg_range)';
    sample_std = smooth(data_std_list(i,:), avg_range)';

    hold on
    curve1 = sample_mean - sample_std;
    curve2 = sample_mean + sample_std;
    inBetween = [curve1, fliplr(curve2)];
%     fill([x_tmp fliplr(x_tmp)],inBetween, color(i), 'LineStyle', 'None', 'FaceAlpha', 0.35);
end


for i=selected
    sample_mean = smooth(data_mean_list(i,:), avg_range)';
    sample_std = smooth(data_std_list(i,:), avg_range)';
    
    hold on;
    plot(sample_mean, color(i), "LineWidth", 2);
end

xlabel("train episode")
ylabel("reward")
legend(data_legend(selected), "location", "southeast")

%%
% plt_list = ["total", "selected"];
% 
% for j = 1:1:2
%     plt = figure(j);
%     exportgraphics(plt,plt_list(j) +'.eps')
% end
