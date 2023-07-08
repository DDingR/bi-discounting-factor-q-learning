clear
close all
% %%
% TRAIN_NAME = "test_train9";
% 
% root_dir = "onnx/" + TRAIN_NAME;
% root_info = dir(root_dir);
% root_num = length(root_info);
% 
% %% data_ori (1000 70)
% onnx_list = [];   
% for j = 3:root_num
%     tmp = root_dir + "/" + root_info(j).name;
%     tmp = table2array(tmp);
%     tmp = tmp(:,1);
%     data_ori = [data_ori tmp];
% end

%%

TRAIN_NAME = "test_train9";

NN_NAME_LIST = [
    "0_1_0"
    "0_5_0"
    ];

%%
pst = ["start" "end"];

for j = 1:1:size(NN_NAME_LIST, 1)

    NN_NAME =  "onnx/" + TRAIN_NAME + "/" + NN_NAME_LIST(j, 1);

    for k = 1:1:2

    nn_full = "./" + NN_NAME + "_" + pst(k) + ".onnx";
    nn_full = importONNXNetwork( ...
    nn_full,  TargetNetwork="dlnetwork", InputDataFormats="BC", OutputDataFormats="BC" ...
    );
    % analyzeNetwork(nn)  
    
    weights = nn_full.Learnables.Value;
    
    [node_num, input_num] = size(weights{1});
    output_num = size(weights{end}, 1);
    layer_num = length(weights);
    
    weight_mat = [
        weights{1} weights{2} weights{3} weights{4} weights{5}' [weights{6}; zeros(node_num-output_num, 1)]
    ];
    weight_mat = extractdata(weight_mat);
    
    figure(j)
    subplot(2,1,k)
    image(weight_mat, "CDataMapping","scaled")
    colormap(jet(256));
    colorbar
%     clim([-0.5 0.5])
    title(pst(k))

    end

end

%%

for j = 1:1:size(NN_NAME_LIST, 1)
    saveas(figure(j), "weight" + string(j) + ".png")
end

%%

% surf(weight_mat(:,20:end))
% view([0,0,1])
% colormap(jet(256));
% colorbar;
% % caxis([-1, 1]);
% xlim([1,size(weight_mat, 2)])
% ylim([1,size(weight_mat, 1)])

%     {128×3   dlarray}
%     {128×1   dlarray}
%     {128×128 dlarray}
%     {128×1   dlarray}
%     { 41×128 dlarray}
%     { 41×1   dlarray}
