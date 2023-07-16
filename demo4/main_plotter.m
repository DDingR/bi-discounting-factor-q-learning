function r_sum =  main_plotter(data)

plot_names = data.plot_names;
SAVE_PLOT = data.SAVE_PLOT;
r_sum = data.r_sum;
data_legend = data.data_legend;
traj_list = data.traj_list;
selected0 = data.selected0;
selected1 = data.selected1;
selected2 = data.selected2;
selected3 = data.selected3;
u_list = data.u_list;

%% PLOT
disp(r_sum)

% % ALL PLOT ========================================
% figure(1)
% tiledlayout(2,1);
% nexttile
% for j = 1:1:case_num
%     plot(traj_list((1) + 2*(j-1), :))
%     hold on
% end
% title("\theta traj", 'fontsize',11,'fontname', 'Times New Roman')
% grid on
% 
% % figure(2)
% nexttile
% for j = 1:1:case_num
%     plot(u_list(j, :))
%     hold on
% end
% title("input", 'fontsize',11,'fontname', 'Times New Roman')
% lgd = legend(data_legend, ...
%     'fontsize',11,'fontname', 'Times New Roman');
% lgd.Layout.Tile = 'south';
% lgd.NumColumns = 3;
% grid on
% % sgtitle(plot_names(1));

% SELECTED0 PLOT ====================================
figure(1)
tiledlayout(2,1);
nexttile
for j = selected0
    plot(traj_list((1) + 2*(j-1), :))
    hold on
end
title("\theta traj", 'fontsize',11,'fontname', 'Times New Roman')
grid on

nexttile
for j = selected0
    plot(u_list(j, :))
    hold on
end
title("input", 'fontsize',11,'fontname', 'Times New Roman')
lgd = legend(data_legend(selected0), ...
    'fontsize',11,'fontname', 'Times New Roman');
lgd.Layout.Tile = 'south';
lgd.NumColumns = 3;
grid on
% sgtitle(plot_names(2));

% SELECTED1 PLOT ====================================
figure(2)
tiledlayout(2,1);
nexttile
for j = selected1
    plot(traj_list((1) + 2*(j-1), :))
    hold on
end
title("\theta traj", 'fontsize',11,'fontname', 'Times New Roman')
grid on

nexttile
for j = selected1
    plot(u_list(j, :))
    hold on
end
title("input", 'fontsize',11,'fontname', 'Times New Roman')
% ONLY FOR THIS CASE
selected1 = [1 16 5 6 7 8];
lgd = legend(data_legend(selected1), ...
    'fontsize',11,'fontname', 'Times New Roman');
lgd.Layout.Tile = 'south';
lgd.NumColumns = 3;
grid on
% sgtitle(plot_names(2));

% SELECTED2 PLOT ====================================
figure(3)
tiledlayout(2,1);
nexttile
for j = selected2
    plot(traj_list((1) + 2*(j-1), :))
    hold on
end
title("\theta traj", 'fontsize',11,'fontname', 'Times New Roman')
grid on

nexttile
for j = selected2
    plot(u_list(j, :))
    hold on
end
title("input", 'fontsize',11,'fontname', 'Times New Roman')
lgd = legend(data_legend(selected2), ...
    'fontsize',11,'fontname', 'Times New Roman');
lgd.Layout.Tile = 'south';
lgd.NumColumns = 3;
grid on
% sgtitle(plot_names(3));

% SELECTED3 PLOT ====================================
figure(4)
tiledlayout(2,1);
nexttile
for j = selected3
    plot(traj_list((1) + 2*(j-1), :))
    hold on
end
title("\theta traj", 'fontsize',11,'fontname', 'Times New Roman')
grid on

nexttile
for j = selected3
    plot(u_list(j, :))
    hold on
end
title("input", 'fontsize',11,'fontname', 'Times New Roman')
lgd = legend(data_legend(selected3), ...
    'fontsize',11,'fontname', 'Times New Roman');
lgd.Layout.Tile = 'south';
lgd.NumColumns = 3;
grid on
% sgtitle(plot_names(4));

%% SAVE PLOTS IN CERTAIN FORMAT
if SAVE_PLOT
%     for j = 1:1:length(plot_names)
%         saveas(figure(j), "fig/" + plot_names(j) + ".fig")
%     end

    for j = 1:1:length(plot_names)
        plt = figure(j);
        exportgraphics(plt, "fig/" + plot_names(j) +'.eps')
    end  
end


end