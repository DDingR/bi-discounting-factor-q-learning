clear
clf

data_file_1 = "all 2.txt";
data_file_2 = "eps 2.txt";


data1 = importdata(data_file_1);
data2 = importdata(data_file_2);

plot_range = 500;
avg_range = 100;
%%
% epilson_greedy = data1(:,1);
small_gamma = data1(:,2);
large_gamma = data1(:,3);

epilson_greedy = data2;

small_gamma = small_gamma(1:plot_range);
large_gamma = large_gamma(1:plot_range);
epilson_greedy = epilson_greedy(1:plot_range);
%%
epilson_greedy_smooth = smooth(epilson_greedy, avg_range);
small_gamma_smooth = smooth(small_gamma, avg_range);
large_gamma_smooth = smooth(large_gamma, avg_range);

%%
data = [epilson_greedy, small_gamma, large_gamma];
data_smooth = [epilson_greedy_smooth small_gamma_smooth large_gamma_smooth];
color = ['r','k','b'];
data_legend = ["epsilon greedy", "small gamma", "large gamma"];

%%
for j = 1:1:3
    figure(j)
    plot(data(:,j), color(j))
    xlabel("train episode")
    ylabel("reward") 
    xlim([0 plot_range])
    ylim([-2000 100])
    legend(data_legend(j), "location", "southeast")
end

figure(4)
for j = 1:1:3
    plot(data_smooth(:,j), color(j))
    hold on
end
xlabel("train episode")
ylabel("reward")
legend(data_legend, "location", "southeast")

%%
plt_list = ["eps", "small", "large", "smooth"];

for j = 1:1:4
    plt = figure(j);
%     saveas(plt, plt_list(j) + '.png')
%     imwrite()
    exportgraphics(plt,plt_list(j) +'.eps')
end

%%

data_smooth(end,:)