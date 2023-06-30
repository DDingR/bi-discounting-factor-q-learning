clear
close all
%%
TRAIN_NAME = "test_train1";

root_dir = "results/" + TRAIN_NAME;
root_info = dir(root_dir);
root_num = length(root_info);

%%
data_ori = [];
for j = 3:root_num
    tmp = readtable(root_dir + "/" + root_info(j).name);
    tmp = table2array(tmp);
    tmp = tmp(:,1);
    data_ori = [data_ori tmp];
end

rep_num = 1;
[episode_num, data_num] = size(data_ori);
case_num = data_num / rep_num;

data = data_ori;

% 
% data = zeros(episode_num, case_num);
% for k = 1:1:case_num
%     data(:,k) = mean(data_ori(:,(1:rep_num-1)+rep_num*(k-1)), 2);
% end

% data = [mean(data(:,1:5),2) mean(data(:,11:15),2) mean(data(:,21:25),2) mean(data(:,31:35),2)]% mean(data(:,41:45),2) mean(data(:,51:55),2)]
% data = [data(:,1) data(:,13) data(:,21) data(:,31)]

%%
avg_range = 50;
smooth_range = 50;

% color = ['r','k','b', 'g', ''];
data_legend = ["gamma0.1", "gamma0.2", "gamma0.5", "eps-greedy"];
% data_legend = string(1:1:case_num);
%%

data_smooth = zeros(size(data_ori));

for i = 1:size(data_ori,2)
    data_smooth(:,i) = smooth(data_ori(:,i), smooth_range);
end

data_err = data_ori - data_smooth;
data_ori_var = var(data_err);
data_ori_var = reshape(data_ori_var, [],case_num);
data_ori_var = mean(data_ori_var)

for j = 1:1:case_num
    figure(j)
    plot(data(:,j)) %, color(j))
    xlabel("train episode")
    ylabel("reward") 
    xlim([0 episode_num])
    ylim([-2000 100])
    legend(data_legend(j), "location", "southeast")
end

figure(case_num+1)
for j = 1:1:case_num
    plot(smooth(data(:,j), avg_range)) %, color(j))
    hold on
end
xlabel("train episode")
ylabel("reward")
legend(data_legend, "location", "southeast")

%%
plt_list = ["gamma0.1", "gamma0.2", "gamma0.5", "eps-greedy", "tt"];

for j = 1:1:5
    plt = figure(j);
%     saveas(plt, plt_list(j) + '.png')
%     imwrite()
    exportgraphics(plt,plt_list(j) +'.eps')
end

%%

data_smooth(end,:)