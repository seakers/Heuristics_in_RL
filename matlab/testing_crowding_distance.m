%% Testing crowding distance function
clear
close all
clc

%% Main

utopia_mode = false;

PF_1 = [0.1, 0.8; 0.2, 0.75; 0.25, 0.65; 0.4, 0.4; 0.5, 0.35; 0.7, 0.1]; % Assuming both objectives must be minimized
cds_1 = compute_crowding_distances(PF_1, utopia_mode);

%%% Case 1 - new design at the edge of current PF
f_new1 = [0.05, 0.9];
PF_new1 = [PF_1; f_new1];
cds_new1 = compute_crowding_distances(PF_new1, utopia_mode);
r_cd1 = mean(cds_1) - mean(cds_new1);

% Plotting
plot_PF(PF_1, 'Original PF')
plot_PF(PF_new1, 'Augmented PF - Edge Case')

%%% Case 2 - new design within current PF
f_new2 = [0.45, 0.37];
PF_new2 = [PF_1; f_new2];
cds_new2 = compute_crowding_distances(PF_new2, utopia_mode);
r_cd2 = mean(cds_1) - mean(cds_new2);

% Plotting
plot_PF(PF_1, 'Original PF')
plot_PF(PF_new2, 'Augmented PF - Internal Case')

%% Functions
function cds = compute_crowding_distances(current_PF, wrt_utopia)

    if wrt_utopia
        current_PF = [0, 1; current_PF; 1, 0];
    end
    
    n_objs = size(current_PF,2);
    cds_objs = zeros(size(current_PF,1), n_objs);
        
    for i = 1:size(current_PF,2)
        objs = current_PF(:,i);
        [~, asc_inds] = sort(objs);
        desc_inds = flip(asc_inds);
        objs_sort = objs(desc_inds);
        
        obj_cds = zeros(size(objs,1), 1);
        objs_max = max(objs);
        objs_min = min(objs);

        obj_cds(1) = 10000; % assign randomly large value
        obj_cds(end) = 10000; % assign randomly large value
        
        for j = 2:size(obj_cds,1)-1
            obj_cds(j) = (objs_sort(j-1) - objs_sort(j+1))/(objs_max - objs_min);
        end
        
        for j = 1:size(objs,1)
            cds_objs(desc_inds(j),i) = obj_cds(j);
        end
    end
    
    cds = sum(cds_objs,2);
end

function [] = plot_PF(current_PF, plot_title)
    figure
    plot(current_PF(:,1), current_PF(:,2),'*')
    xlabel('$f_1(x)$','Interpreter','Latex')
    ylabel('$f_2(x)$','Interpreter','Latex')
    title(plot_title)
end