% WES PORTER
% 7/31/2017
% USDA PROJECT - Preparing secondary climate files using a nearest 
% neighbor approach.

% GHCND data
% Read Arkansas data
climateArk = dlmread('Ark_misc_v1.csv',',',1,1); 

% Read Bolivar_v1.csv
climateBol = dlmread('Bolivar_v1.csv',',',1,1);

% Read Coahoma_v1.csv
climateCoa = dlmread('Coahoma_v1.csv',',',1,1);

% Read Quitman_v1.csv
climateQui = dlmread('Quitman_v1.csv',',',1,1);

% Read Sunflower_v1.csv
climateSun = dlmread('Sunflower_v1.csv',',',1,1);

% Read Tallahatchie_v1.csv
climateTal = dlmread('Tallahatchie_v1.csv',',',1,1);

% Read Tunica_v1.csv
climateTun = dlmread('Tunica_v1.csv',',',1,1);

% Subset the GHCND data based on the climate stations

%Arkansas
Helena30 = climateArk(climateArk(:,1)==1,:); % Helena 3.0 SSW

Helena = climateArk(climateArk(:,1)==2,:); % Helena

LakeView = climateArk(climateArk(:,1)==4,:); % Lake View

HuxtablePP = climateArk(climateArk(:,1)==9,:); % W G Huxtable Pumping Plant

%Bolivar
Cleveland3 = climateBol(climateBol(:,1)==1,:); % Cleveland3 N

Cleveland = climateBol(climateBol(:,1)==2,:); % Cleveland

Duncan01 = climateBol(climateBol(:,1)==3,:); %Duncan0.1 NNW

Duncan = climateBol(climateBol(:,1)==4,:); %Duncan

Rosedale = climateBol(climateBol(:,1)==5,:); %Rosedale

Shaw01 = climateBol(climateBol(:,1)==7,:); %Shaw 0.1 WSW

% Coahoma
Clark29 = climateCoa(climateCoa(:,1)==1,:); % Clarksdale 2.9 SW

Clark = climateCoa(climateCoa(:,1)==2,:); % Clarksdale (Majority of data)

% Quitman
Lambert1 = climateQui(climateQui(:,1)==1,:); % Lambert 1 W

Vance2 = climateQui(climateQui(:,1)==3,:); % Vance 2 

%Sunflower
Indianola = climateSun(climateSun(:,1)==1,:); % Indianola 1.1 N

Moorhead = climateSun(climateSun(:,1)==2,:); % Moorhead

Sunflower = climateSun(climateSun(:,1)==3,:); % Sunflower 3.9 N

% Tallahatchie
Charleston35 = climateTal(climateTal(:,1)==1,:); % Charleston 3.5 SW

Charleston = climateTal(climateTal(:,1)==2,:); % Charleston

MinterCity = climateTal(climateTal(:,1)==3,:); % Minter City

SwanLake = climateTal(climateTal(:,1)==4,:); % Swan Lake

Vance1 = climateTal(climateTal(:,1)==5,:); % Vance 1 SW

% Tunica
Robinsonville54 = climateTun(climateTun(:,1)==1,:); % Robinsonville 5.4 E

Sarah3 = climateTun(climateTun(:,1)==2,:); % Sarah 3 SE

Tunica2 = climateTun(climateTun(:,1)==3,:); % Tunica 2 N

% SCAN Site data

BeasleyLake = dlmread('Beasley_Lake_daily_v1.csv',',',1,1);

Perthshire = dlmread('Perthshire_daily_v1.csv',',',1,1);

SandyRidge = dlmread('Sandy_Ridge_daily_v1.csv',',',1,1);

Tunica = dlmread('Tunica_daily_v1.csv',',',1,1);

Vance = dlmread('Vance_daily_v1.csv',',',1,1); 

% Complete list of all dates since 1970 Jan 1
dates = dlmread('Dates.csv',',');
total_num_days = length(dates);

%% 
% BUILDING SECONDARY CLIMATE FILES
% Loop through each column in climate
% Includes PRCP,SNWD,SNOW,TMAX,TMIN,TOBS

% HELENA 3.0 SSW
%Create climate output files where aggregated data will go
climateOut_Helena30 = zeros(length(dates),5);
climateOut_Helena30(:,1) = dates;

fillHelena30 = 0;
fillHelena30_prcp = 0;
fillHelena = 0;
fillHelena_prcp = 0;
fillLakeView = 0;
fillLakeView_prcp = 0;
fillHuxtablePP = 0;
fillHuxtablePP_prcp = 0;
fillTunica = 0;
fillTunica_prcp = 0;

for j = 3:8


    % Populating climate data from Helena 3.0 SSW first
    station = Helena30; 

    %Check to see if station includes data before 1 Jan 1970
    % If it does, exclude that data

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    %Initialize for loop
    count = 1;
    index = 1;
 
    % If the station has data, append it for the respective date 
    % to the climateOut_Helena30 array
    for i = 1:length(dates)
        if count < length(station) +1
            if dates(i,1) == station(count,2)
                climateOut_Helena30(index,j-1) = station(count,j);
                if j == 3 && station(count,j) ~= -9999
                    fillHelena30_prcp = fillHelena30_prcp +1;
                end
                count = count + 1;
                index = index + 1;
                fillHelena30 = fillHelena30 +1; 
            else
                climateOut_Helena30(index,j-1) = -9999;
                index = index + 1;
            end
        else
            climateOut_Helena30(index,j-1) = -9999;
            index = index + 1;
        end             
    end

    % Add Helena to fill in no data gaps
    station = Helena;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Helena30(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Helena30(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillHelena = fillHelena +1; % How many data points were added
                    if j == 3
                    fillHelena_prcp = fillHelena_prcp +1;
                    end
                else count = count +1;
                end
            end 
        end
    end

    % Add Lake View to fill in no data gaps
    station = LakeView;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Helena30(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Helena30(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillLakeView = fillLakeView +1; % How many data points were added
                    if j == 3
                    fillLakeView_prcp = fillLakeView_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end    
    
    % Add W G Huxtable Pumping Plant to fill in no data gaps
    station = HuxtablePP;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Helena30(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Helena30(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillHuxtablePP = fillHuxtablePP +1; % How many data points were added
                    if j == 3
                    fillHuxtablePP_prcp = fillHuxtablePP_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end 
    
    % Add Tunica SCAN Site to fill in no data gaps
    station = Tunica;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Helena30(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Helena30(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillTunica = fillTunica +1; % How many data points were added
                    if j == 3
                    fillTunica_prcp = fillTunica_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
end

% Determine number of no data remaining after aggregation.
noDat_Helena30 = sum(climateOut_Helena30(:,2)==-9999);
tmp = climateOut_Helena30(climateOut_Helena30(:,1) >= 20000101,:);
noDat_Helena30_subset = sum(tmp(:,2)==-9999);

perc_prcp_Helena30 = (fillHelena30_prcp/total_num_days)*100;
perc_prcp_Helena = (fillHelena_prcp/total_num_days)*100;
perc_prcp_LakeView = (fillLakeView_prcp/total_num_days)*100;
perc_prcp_HuxtablePP = (fillHuxtablePP_prcp/total_num_days)*100;
perc_prcp_Tunica = (fillTunica_prcp/total_num_days)*100;

% Create precipitation stats table
stats_Helena30 = table(categorical({'Helena 3.0 SSW';'Helena';'Lake View';'W G Huxtable Pumping Plant';'Tunica SCAN Site'}),...
[perc_prcp_Helena30;perc_prcp_Helena;perc_prcp_LakeView;perc_prcp_HuxtablePP;perc_prcp_Tunica],...
'VariableNames',{'Station' 'PercentFilled'});

% Export climate data
T = array2table(climateOut_Helena30,'VariableNames',{'DATE','PRCP','SNWD','SNOW','TMAX','TMIN','TOBS'});
writetable(T,'Output_Helena30.csv')

% Export stats
writetable(stats_Helena30,'Stats_Helena30.csv')

%%
% CLARKSDALE
%Create climate output files where aggregated data will go
climateOut_Clarksdale = zeros(length(dates),5);
climateOut_Clarksdale(:,1) = dates;

fillClark = 0;
fillClark_prcp = 0;
fillClark29 = 0;
fillClark29_prcp = 0;
fillDuncan = 0;
fillDuncan_prcp = 0;
fillDuncan01 = 0;
fillDuncan01_prcp = 0;
fillVance2 = 0;
fillVance2_prcp = 0;
fillVance = 0;
fillVance_prcp = 0;
fillVance1 = 0;
fillVance1_prcp = 0;

for j = 3:8

    % Populating climate data from Clarksdale first
    station = Clark; 

    %Check to see if station includes data before 1 Jan 1970
    % If it does, exclude that data

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    %Initialize for loop
    count = 1;
    index = 1;
 
    % If the station has data, append it for the respective date 
    % to the climateOut_Clarksdale array
    for i = 1:length(dates)
        if count < length(station) +1
            if dates(i,1) == station(count,2)
                climateOut_Clarksdale(index,j-1) = station(count,j);
                if j == 3 && station(count,j) ~= -9999
                    fillClark_prcp = fillClark_prcp +1;
                end
                count = count + 1;
                index = index + 1;
                fillClark = fillClark +1; 
            else
                climateOut_Clarksdale(index,j-1) = -9999;
                index = index + 1;
            end
        else
            climateOut_Clarksdale(index,j-1) = -9999;
            index = index + 1;
        end             
    end

    % Add Clarksdale 2.9 SW to fill in no data gaps
    station = Clark29;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Clarksdale(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Clarksdale(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillClark29 = fillClark29 +1; % How many data points were added
                    if j == 3
                    fillClark29_prcp = fillClark29_prcp +1;
                    end
                else count = count +1;
                end
            end 
        end
    end

    % Add Duncan to fill in no data gaps
    station = Duncan;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Clarksdale(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Clarksdale(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillDuncan = fillDuncan +1; % How many data points were added
                    if j == 3
                    fillDuncan_prcp = fillDuncan_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end    
    
    % Add Duncan 0.1 NNW to fill in no data gaps
    station = Duncan01;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Clarksdale(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Clarksdale(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillDuncan01 = fillDuncan01 +1; % How many data points were added
                    if j == 3
                    fillDuncan01_prcp = fillDuncan01_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end 
    
    % Add Vance 2 to fill in no data gaps
    station = Vance2;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Clarksdale(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Clarksdale(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillVance2 = fillVance2 +1; % How many data points were added
                    if j == 3
                    fillVance2_prcp = fillVance2_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Vance SCAN Site to fill in no data gaps
    station = Vance;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Clarksdale(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Clarksdale(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillVance = fillVance +1; % How many data points were added
                    if j == 3
                    fillVance_prcp = fillVance_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Vance 1 SW to fill in no data gaps
    station = Vance1;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Clarksdale(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Clarksdale(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillVance1 = fillVance1 +1; % How many data points were added
                    if j == 3
                    fillVance1_prcp = fillVance1_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
end

% Determine number of no data remaining after aggregation.
noDat_Clarksdale = sum(climateOut_Clarksdale(:,2)== -9999);
tmp = climateOut_Clarksdale(climateOut_Clarksdale(:,1) >= 20000101,:);
noDat_Clarksdale_subset = sum(tmp(:,2)==-9999);

perc_prcp_Clark = (fillClark_prcp/total_num_days)*100;
perc_prcp_Clark29 = (fillClark29_prcp/total_num_days)*100;
perc_prcp_Duncan = (fillDuncan_prcp/total_num_days)*100;
perc_prcp_Duncan01 = (fillDuncan01_prcp/total_num_days)*100;
perc_prcp_Vance2 = (fillVance2_prcp/total_num_days)*100;
perc_prcp_Vance = (fillVance_prcp/total_num_days)*100;
perc_prcp_Vance1 = (fillVance1_prcp/total_num_days)*100;

% Create stats table
stats_Clarksdale = table(categorical({'Clarksdale';'Clarksdale 2.9 SW';'Duncan';'Duncan 0.1 NNW';'Vance 2';'Vance SCAN Site';'Vance 1 SW'}),...
    [perc_prcp_Clark;perc_prcp_Clark29;perc_prcp_Duncan;perc_prcp_Duncan01;perc_prcp_Vance2;perc_prcp_Vance;perc_prcp_Vance1],...
    'VariableNames',{'Station' 'PercentFilled'});

% Export climate data
T = array2table(climateOut_Clarksdale,'VariableNames',{'DATE','PRCP','SNWD','SNOW','TMAX','TMIN','TOBS'});
writetable(T,'Output_Clarksdale.csv')

% Export stats
writetable(stats_Clarksdale,'Stats_Clarksdale.csv')

%%
% VANCE 1 SW
%Create climate output files where aggregated data will go
climateOut_Vance1 = zeros(length(dates),5);
climateOut_Vance1(:,1) = dates;

fillVance1 = 0;
fillVance1_prcp = 0;
fillVance = 0;
fillVance_prcp = 0;
fillVance2 = 0;
fillVance2_prcp = 0;
fillLambert1 = 0;
fillLambert1_prcp = 0;
fillSwanLake = 0;
fillSwanLake_prcp =0;
fillClark = 0;
fillClark_prcp = 0;
fillCharleston35 = 0;
fillCharleston35_prcp = 0;
fillClark29 = 0;
fillClark29_prcp = 0;
fillCharleston = 0;
fillCharleston_prcp = 0;

for j = 3:8

    % Populating climate data from Vance 1 SW first
    station = Vance1; 

    %Check to see if station includes data before 1 Jan 1970
    % If it does, exclude that data

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    %Initialize for loop
    count = 1;
    index = 1;
 
    % If the station has data, append it for the respective date 
    % to the climateOut_Vance1 array
    for i = 1:length(dates)
        if count < length(station) +1
            if dates(i,1) == station(count,2)
                climateOut_Vance1(index,j-1) = station(count,j);
                if j == 3 && station(count,j) ~= -9999
                    fillVance1_prcp = fillVance1_prcp +1;
                end
                count = count + 1;
                index = index + 1;
                fillVance1 = fillVance1 +1; 
            else
                climateOut_Vance1(index,j-1) = -9999;
                index = index + 1;
            end
        else
            climateOut_Vance1(index,j-1) = -9999;
            index = index + 1;
        end             
    end


    % Add Vance SCAN Site to fill in no data gaps
    station = Vance;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Vance1(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Vance1(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillVance = fillVance +1; % How many data points were added
                    if j == 3
                    fillVance_prcp = fillVance_prcp +1;
                    end
                else count = count +1;
                end
            end 
        end
    end

    % Add Vance 2 to fill in no data gaps
    station = Vance2;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Vance1(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Vance1(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillVance2 = fillVance2 +1; % How many data points were added
                    if j == 3
                    fillVance2_prcp = fillVance2_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end    
    
    % Add Lambert 1 W to fill in no data gaps
    station = Lambert1;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Vance1(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Vance1(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillLambert1 = fillLambert1 +1; % How many data points were added
                    if j == 3
                    fillLambert1_prcp = fillLambert1_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end 
    
    % Add Swan Lake to fill in no data gaps
    station = SwanLake;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Vance1(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Vance1(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillSwanLake = fillSwanLake +1; % How many data points were added
                    if j == 3
                    fillSwanLake_prcp = fillSwanLake_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Clarksdale to fill in no data gaps
    station = Clark;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Vance1(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Vance1(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillClark = fillClark +1; % How many data points were added
                    if j == 3
                    fillClark_prcp = fillClark_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Charleston 3.5 SW to fill in no data gaps
    station = Charleston35;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Vance1(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Vance1(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillCharleston35 = fillCharleston35 +1; % How many data points were added
                    if j == 3
                    fillCharleston35_prcp = fillCharleston35_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Clarksdale 2.9 SW to fill in no data gaps
    station = Clark29;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Vance1(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Vance1(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillClark29 = fillClark29 +1; % How many data points were added
                    if j == 3
                    fillClark29_prcp = fillClark29_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Charleston to fill in no data gaps
    station = Charleston;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Vance1(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Vance1(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillCharleston = fillCharleston +1; % How many data points were added
                    if j == 3
                    fillCharleston_prcp = fillCharleston_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
end

% Determine number of no data remaining after aggregation.
noDat_Vance1 = sum(climateOut_Vance1(:,2)== -9999);
tmp = climateOut_Vance1(climateOut_Vance1(:,1) >= 20000101,:);
noDat_Vance1_subset = sum(tmp(:,2)==-9999);

perc_prcp_Vance1 = (fillVance1_prcp/total_num_days)*100;
perc_prcp_Vance = (fillVance_prcp/total_num_days)*100;
perc_prcp_Vance2 = (fillVance2_prcp/total_num_days)*100;
perc_prcp_Lambert1 = (fillLambert1_prcp/total_num_days)*100;
perc_prcp_SwanLake = (fillSwanLake_prcp/total_num_days)*100;
perc_prcp_Clark = (fillClark_prcp/total_num_days)*100;
perc_prcp_Charleston35 = (fillCharleston35_prcp/total_num_days)*100;
perc_prcp_Clark29 = (fillClark29_prcp/total_num_days)*100;
perc_prcp_Charleston = (fillCharleston_prcp/total_num_days)*100;

% Create stats table
stats_Vance1 = table(categorical({'Vance 1 SW';'Vance SCAN Site';'Vance 2';'Lambert 1 W';'Swan Lake';'Clarksdale';'Charleston 3.5 SW';'Clarksdale 2.9 SW';'Charleston'}),...
    [perc_prcp_Vance1;perc_prcp_Vance;perc_prcp_Vance2;perc_prcp_Lambert1;perc_prcp_SwanLake;perc_prcp_Clark;perc_prcp_Charleston35;perc_prcp_Clark29;perc_prcp_Charleston],...
    'VariableNames',{'Station' 'PercentFilled'});

% Export climate data
T = array2table(climateOut_Vance1,'VariableNames',{'DATE','PRCP','SNWD','SNOW','TMAX','TMIN','TOBS'});
writetable(T,'Output_Vance1.csv')

% Export stats
writetable(stats_Vance1,'Stats_Vance1.csv')

%%
% Duncan 0.1 NNW
%Create climate output files where aggregated data will go
climateOut_Duncan01 = zeros(length(dates),5);
climateOut_Duncan01(:,1) = dates;

fillDuncan01 = 0;
fillDuncan01_prcp = 0;
fillDuncan = 0;
fillDuncan_prcp = 0;
fillPerthshire = 0;
fillPerthshire_prcp = 0;
fillClark29 = 0;
fillClark29_prcp = 0;
fillClark = 0;
fillClark_prcp = 0;
fillCleveland3 = 0;
fillCleveland3_prcp = 0;

for j = 3:8


    % Populating climate data from Duncan 0.1 NNW first
    station = Duncan01; 

    %Check to see if station includes data before 1 Jan 1970
    % If it does, exclude that data

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    %Initialize for loop
    count = 1;
    index = 1;
 
    % If the station has data, append it for the respective date 
    % to the climateOut_Duncan01 array
    for i = 1:length(dates)
        if count < length(station) +1
            if dates(i,1) == station(count,2)
                climateOut_Duncan01(index,j-1) = station(count,j);
                if j == 3 && station(count,j) ~= -9999
                    fillDuncan01_prcp = fillDuncan01_prcp +1;
                end
                count = count + 1;
                index = index + 1;
                fillDuncan01 = fillDuncan01 +1; 
            else
                climateOut_Duncan01(index,j-1) = -9999;
                index = index + 1;
            end
        else
            climateOut_Duncan01(index,j-1) = -9999;
            index = index + 1;
        end             
    end

    % Add Duncan to fill in no data gaps
    station = Duncan;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Duncan01(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Duncan01(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillDuncan = fillDuncan +1; % How many data points were added
                    if j == 3
                    fillDuncan_prcp = fillDuncan_prcp +1;
                    end
                else count = count +1;
                end
            end 
        end
    end

    % Add Perthshire SCAN Site to fill in no data gaps
    station = Perthshire;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Duncan01(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Duncan01(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillPerthshire = fillPerthshire +1; % How many data points were added
                    if j == 3
                    fillPerthshire_prcp = fillPerthshire_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end    
    
    % Add Clarksdale 2.9 SW to fill in no data gaps
    station = Clark29;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Duncan01(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Duncan01(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillClark29 = fillClark29 +1; % How many data points were added
                    if j == 3
                    fillClark29_prcp = fillClark29_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end 
    
    % Add Clarksdale to fill in no data gaps
    station = Clark;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Duncan01(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Duncan01(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillClark = fillClark +1; % How many data points were added
                    if j == 3
                    fillClark_prcp = fillClark_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Cleveland 3 N to fill in no data gaps
    station = Cleveland3;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Duncan01(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Duncan01(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillCleveland3 = fillCleveland3 +1; % How many data points were added
                    if j == 3
                    fillCleveland3_prcp = fillCleveland3_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
end

% Determine number of no data remaining after aggregation.
noDat_Duncan01 = sum(climateOut_Duncan01(:,2)== -9999);
tmp = climateOut_Duncan01(climateOut_Duncan01(:,1) >= 20000101,:);
noDat_Duncan01_subset = sum(tmp(:,2)==-9999);


perc_prcp_Duncan01 = (fillDuncan01_prcp/total_num_days)*100;
perc_prcp_Duncan = (fillDuncan_prcp/total_num_days)*100;
perc_prcp_Perthshire = (fillPerthshire_prcp/total_num_days)*100;
perc_prcp_Clark29 = (fillClark29_prcp/total_num_days)*100;
perc_prcp_Clark = (fillClark_prcp/total_num_days)*100;
perc_prcp_Cleveland3 = (fillCleveland3_prcp/total_num_days)*100;

% Create stats table
stats_Duncan01 = table(categorical({'Duncan 0.1 NNW';'Duncan';'Perthshire SCAN Site';'Clarksdale 2.9 SW';'Clarksdale';'Cleveland 3 N'}),...
    [perc_prcp_Duncan01;perc_prcp_Duncan;perc_prcp_Perthshire;perc_prcp_Clark29;perc_prcp_Clark;perc_prcp_Cleveland3],...
    'VariableNames',{'Station' 'PercentFilled'});

% Export climate data
T = array2table(climateOut_Duncan01,'VariableNames',{'DATE','PRCP','SNWD','SNOW','TMAX','TMIN','TOBS'});
writetable(T,'Output_Duncan01.csv')

% Export stats
writetable(stats_Duncan01,'Stats_Duncan01.csv')

%%
% Cleveland 3 N
%Create climate output files where aggregated data will go
climateOut_Cleveland3 = zeros(length(dates),5);
climateOut_Cleveland3(:,1) = dates;

fillCleveland3 = 0;
fillCleveland3_prcp = 0;
fillCleveland = 0;
fillCleveland_prcp = 0;
fillSandyRidge = 0;
fillSandyRidge_prcp = 0;
fillShaw01 = 0;
fillShaw01_prcp = 0;
fillPerthshire = 0;
fillPerthshire_prcp = 0;
fillSunflower = 0;
fillSunflower_prcp = 0;
fillDuncan01 = 0;
fillDuncan01_prcp = 0;
fillDuncan = 0;
fillDuncan_prcp = 0;
fillRosedale = 0;
fillRosedale_prcp = 0;

for j = 3:8


    % Populating climate data from Cleveland 3 N first
    station = Cleveland3; 

    %Check to see if station includes data before 1 Jan 1970
    % If it does, exclude that data

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    %Initialize for loop
    count = 1;
    index = 1;
 
    % If the station has data, append it for the respective date 
    % to the climateOut_Cleveland3 array
    for i = 1:length(dates)
        if count < length(station) +1
            if dates(i,1) == station(count,2)
                climateOut_Cleveland3(index,j-1) = station(count,j);
                if j == 3 && station(count,j) ~= -9999
                    fillCleveland3_prcp = fillCleveland3_prcp +1;
                end
                count = count + 1;
                index = index + 1;
                fillCleveland3 = fillCleveland3 +1; 
            else
                climateOut_Cleveland3(index,j-1) = -9999;
                index = index + 1;
            end
        else
            climateOut_Cleveland3(index,j-1) = -9999;
            index = index + 1;
        end             
    end

    % Add Cleveland to fill in no data gaps
    station = Cleveland;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Cleveland3(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Cleveland3(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillCleveland = fillCleveland +1; % How many data points were added
                    if j == 3
                    fillCleveland_prcp = fillCleveland_prcp +1;
                    end
                else count = count +1;
                end
            end 
        end
    end
    
    % Add Sandy Ridge SCAN Site to fill in no data gaps
    station = SandyRidge;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Cleveland3(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Cleveland3(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillSandyRidge = fillSandyRidge +1; % How many data points were added
                    if j == 3
                    fillSandyRidge_prcp = fillSandyRidge_prcp +1;
                    end
                else count = count +1;
                end
            end 
        end
    end
    
    % Add Shaw 0.1 WSW to fill in no data gaps
    station = Shaw01;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Cleveland3(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Cleveland3(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillShaw01 = fillShaw01 +1; % How many data points were added
                    if j == 3
                    fillShaw01_prcp = fillShaw01_prcp +1;
                    end
                else count = count +1;
                end
            end 
        end
    end
    
    % Add Perthshire SCAN Site to fill in no data gaps
    station = Perthshire;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Cleveland3(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Cleveland3(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillPerthshire = fillPerthshire +1; % How many data points were added
                    if j == 3
                    fillPerthshire_prcp = fillPerthshire_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Sunflower 3.9 N to fill in no data gaps
    station = Sunflower;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Cleveland3(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Cleveland3(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillSunflower = fillSunflower +1; % How many data points were added
                    if j == 3
                    fillSunflower_prcp = fillSunflower_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end 
    
    % Add Duncan 0.1 NNW to fill in no data gaps
    station = Duncan01;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Cleveland3(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Cleveland3(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillDuncan01 = fillDuncan01 +1; % How many data points were added
                    if j == 3
                    fillDuncan01_prcp = fillDuncan01_prcp +1;
                    end
                else count = count +1;
                end
            end 
        end
    end  
    
    % Add Duncan to fill in no data gaps
    station = Duncan;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Cleveland3(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Cleveland3(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillDuncan = fillDuncan +1; % How many data points were added
                    if j == 3
                    fillDuncan_prcp = fillDuncan_prcp +1;
                    end
                else count = count +1;
                end
            end 
        end
    end 
    
    % Add Rosedale to fill in no data gaps
    station = Rosedale;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Cleveland3(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Cleveland3(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillRosedale = fillRosedale +1; % How many data points were added
                    if j == 3
                    fillRosedale_prcp = fillRosedale_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
end

% Determine number of no data remaining after aggregation.
noDat_Cleveland3 = sum(climateOut_Cleveland3(:,2)== -9999);
tmp = climateOut_Cleveland3(climateOut_Cleveland3(:,1) >= 20000101,:);
noDat_Cleveland3_subset = sum(tmp(:,2)==-9999);

perc_prcp_Cleveland3 = (fillCleveland3_prcp/total_num_days)*100;
perc_prcp_Cleveland = (fillCleveland_prcp/total_num_days)*100;
perc_prcp_SandyRidge = (fillSandyRidge_prcp/total_num_days)*100;
perc_prcp_Shaw01 = (fillShaw01_prcp/total_num_days)*100;
perc_prcp_Perthshire = (fillPerthshire_prcp/total_num_days)*100;
perc_prcp_Sunflower = (fillSunflower_prcp/total_num_days)*100;
perc_prcp_Duncan01 = (fillDuncan01_prcp/total_num_days)*100;
perc_prcp_Duncan = (fillDuncan_prcp/total_num_days)*100;
perc_prcp_Rosedale = (fillRosedale_prcp/total_num_days)*100;

% Create stats table
stats_Cleveland3 = table(categorical({'Cleveland 3N';'Cleveland';'Sandy Ridge SCAN Site';'Shaw 0.1 WSW';'Perthshire SCAN Site';'Sunflower 3.9 N';'Duncan 0.1 NNW';'Duncan';'Rosedale'}),...
    [perc_prcp_Cleveland3;perc_prcp_Cleveland;perc_prcp_SandyRidge;perc_prcp_Shaw01;perc_prcp_Perthshire;perc_prcp_Sunflower;perc_prcp_Duncan01;perc_prcp_Duncan;perc_prcp_Rosedale],...
    'VariableNames',{'Station' 'PercentFilled'});

% Export climate data
T = array2table(climateOut_Cleveland3,'VariableNames',{'DATE','PRCP','SNWD','SNOW','TMAX','TMIN','TOBS'});
writetable(T,'Output_Cleveland3.csv')

% Export stats
writetable(stats_Cleveland3,'Stats_Cleveland3.csv')

%%
% Cleveland
%Create climate output files where aggregated data will go
climateOut_Cleveland = zeros(length(dates),5);
climateOut_Cleveland(:,1) = dates;

fillCleveland = 0;
fillCleveland_prcp = 0;
fillCleveland3 = 0;
fillCleveland3_prcp = 0;
fillShaw01 = 0;
fillShaw01_prcp = 0;
fillSandyRidge = 0;
fillSandyRidge_prcp = 0;
fillSunflower = 0;
fillSunflower_prcp = 0;
fillRosedale = 0;
fillRosedale_prcp = 0;
fillPerthshire = 0;
fillPerthshire_prcp = 0;

for j = 3:8

    % Populating climate data from Cleveland first
    station = Cleveland; 

    %Check to see if station includes data before 1 Jan 1970
    % If it does, exclude that data

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    %Initialize for loop
    count = 1;
    index = 1;
 
    % If the station has data, append it for the respective date 
    % to the climateOut_Cleveland array
    for i = 1:length(dates)
        if count < length(station) +1
            if dates(i,1) == station(count,2)
                climateOut_Cleveland(index,j-1) = station(count,j);
                if j == 3 && station(count,j) ~= -9999
                    fillCleveland_prcp = fillCleveland_prcp +1;
                end
                count = count + 1;
                index = index + 1;
                fillCleveland = fillCleveland +1; 
            else
                climateOut_Cleveland(index,j-1) = -9999;
                index = index + 1;
            end
        else
            climateOut_Cleveland(index,j-1) = -9999;
            index = index + 1;
        end             
    end
    
    % Add Cleveland 3 N to fill in no data gaps
    station = Cleveland3;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Cleveland(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Cleveland(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillCleveland3 = fillCleveland3 +1; % How many data points were added
                    if j == 3
                    fillCleveland3_prcp = fillCleveland3_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Shaw 0.1 WSW to fill in no data gaps
    station = Shaw01;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Cleveland(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Cleveland(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillShaw01 = fillShaw01 +1; % How many data points were added
                    if j == 3
                    fillShaw01_prcp = fillShaw01_prcp +1;
                    end
                else count = count +1;
                end
            end 
        end
    end
    
    % Add Sandy Ridge SCAN Site to fill in no data gaps
    station = SandyRidge;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Cleveland(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Cleveland(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillSandyRidge = fillSandyRidge +1; % How many data points were added
                    if j == 3
                    fillSandyRidge_prcp = fillSandyRidge_prcp +1;
                    end
                else count = count +1;
                end
            end 
        end
    end
    
    % Add Sunflower 3.9 N to fill in no data gaps
    station = Sunflower;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Cleveland(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Cleveland(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillSunflower = fillSunflower +1; % How many data points were added
                    if j == 3
                    fillSunflower_prcp = fillSunflower_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end 
    
    % Add Rosedale to fill in no data gaps
    station = Rosedale;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Cleveland(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Cleveland(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillRosedale = fillRosedale +1; % How many data points were added
                    if j == 3
                    fillRosedale_prcp = fillRosedale_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Perthshire SCAN Site to fill in no data gaps
    station = Perthshire;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Cleveland(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Cleveland(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillPerthshire = fillPerthshire +1; % How many data points were added
                    if j == 3
                    fillPerthshire_prcp = fillPerthshire_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end
end

% Determine number of no data remaining after aggregation.
noDat_Cleveland = sum(climateOut_Cleveland(:,2)== -9999);
tmp = climateOut_Cleveland(climateOut_Cleveland(:,1) >= 20000101,:);
noDat_Cleveland_subset = sum(tmp(:,2)==-9999);

perc_prcp_Cleveland = (fillCleveland_prcp/total_num_days)*100;
perc_prcp_Cleveland3 = (fillCleveland3_prcp/total_num_days)*100;
perc_prcp_Shaw01 = (fillShaw01_prcp/total_num_days)*100;
perc_prcp_SandyRidge = (fillSandyRidge_prcp/total_num_days)*100;
perc_prcp_Sunflower = (fillSunflower_prcp/total_num_days)*100;
perc_prcp_Rosedale = (fillRosedale_prcp/total_num_days)*100;
perc_prcp_Perthshire = (fillPerthshire_prcp/total_num_days)*100;

% Create stats table
stats_Cleveland = table(categorical({'Cleveland';'Cleveland 3 N';'Shaw 0.1 WSW';'Sandy Ridge SCAN Site';'Sunflower 3.9 N';'Rosedale';'Perthshire SCAN Site'}),...
    [perc_prcp_Cleveland;perc_prcp_Cleveland3;perc_prcp_Shaw01;perc_prcp_SandyRidge;perc_prcp_Sunflower;perc_prcp_Rosedale;perc_prcp_Perthshire],...
    'VariableNames',{'Station' 'PercentFilled'});

% Export climate data
T = array2table(climateOut_Cleveland,'VariableNames',{'DATE','PRCP','SNWD','SNOW','TMAX','TMIN','TOBS'});
writetable(T,'Output_Cleveland.csv')

% Export stats
writetable(stats_Cleveland,'Stats_Cleveland.csv')


%%
% Sunflower 
%Create climate output files where aggregated data will go
climateOut_Sunflower = zeros(length(dates),5);
climateOut_Sunflower(:,1) = dates;

fillSunflower = 0;
fillSunflower_prcp = 0;
fillSandyRidge = 0;
fillSandyRidge_prcp = 0;
fillMoorhead = 0;
fillMoorhead_prcp = 0;
fillIndianola = 0;
fillIndianola_prcp = 0;
fillShaw01 = 0;
fillShaw01_prcp = 0;
fillCleveland = 0;
fillCleveland_prcp = 0;
fillBeasleyLake = 0;
fillBeasleyLake_prcp = 0;
fillCleveland3 = 0;
fillCleveland3_prcp = 0;
fillMinterCity = 0;
fillMinterCity_prcp = 0;

for j = 3:8

    % Populating climate data from Sunflower 3.9 N first
    station = Sunflower; 

    %Check to see if station includes data before 1 Jan 1970
    % If it does, exclude that data

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    %Initialize for loop
    count = 1;
    index = 1;
 
    % If the station has data, append it for the respective date 
    % to the climateOut_Sunflower array
    for i = 1:length(dates)
        if count < length(station) +1
            if dates(i,1) == station(count,2)
                climateOut_Sunflower(index,j-1) = station(count,j);
                if j == 3 && station(count,j) ~= -9999
                    fillSunflower_prcp = fillSunflower_prcp +1;
                end
                count = count + 1;
                index = index + 1;
                fillSunflower = fillSunflower +1; 
            else
                climateOut_Sunflower(index,j-1) = -9999;
                index = index + 1;
            end
        else
            climateOut_Sunflower(index,j-1) = -9999;
            index = index + 1;
        end             
    end
    
    % Add Sandy Ridge SCAN Site to fill in no data gaps
    station = SandyRidge;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Sunflower(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Sunflower(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillSandyRidge = fillSandyRidge +1; % How many data points were added
                    if j == 3
                    fillSandyRidge_prcp = fillSandyRidge_prcp +1;
                    end
                else count = count +1;
                end
            end 
        end
    end
    
    % Add Moorhead to fill in no data gaps
    station = Moorhead;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Sunflower(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Sunflower(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillMoorhead = fillMoorhead +1; % How many data points were added
                    if j == 3
                    fillMoorhead_prcp = fillMoorhead_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Indianola to fill in no data gaps
    station = Indianola;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Sunflower(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Sunflower(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillIndianola = fillIndianola +1; % How many data points were added
                    if j == 3
                    fillIndianola_prcp = fillIndianola_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Shaw 0.1 WSW to fill in no data gaps
    station = Shaw01;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Sunflower(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Sunflower(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillShaw01 = fillShaw01 +1; % How many data points were added
                    if j == 3
                    fillShaw01_prcp = fillShaw01_prcp +1;
                    end
                else count = count +1;
                end
            end 
        end
    end
    
    % Add Cleveland to fill in no data gaps
    station = Cleveland;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Sunflower(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Sunflower(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillCleveland = fillCleveland +1; % How many data points were added
                    if j == 3
                    fillCleveland_prcp = fillCleveland_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Beasley Lake SCAN Site to fill in no data gaps
    station = BeasleyLake;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Sunflower(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Sunflower(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillBeasleyLake = fillBeasleyLake +1; % How many data points were added
                    if j == 3
                    fillBeasleyLake_prcp = fillBeasleyLake_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Cleveland 3 N to fill in no data gaps
    station = Cleveland3;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Sunflower(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Sunflower(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillCleveland3 = fillCleveland3 +1; % How many data points were added
                    if j == 3
                    fillCleveland3_prcp = fillCleveland3_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    % Add Minter City to fill in no data gaps
    station = MinterCity;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101 & station(:,2) <= 20151231,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2015
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut_Sunflower(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut_Sunflower(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillMinterCity = fillMinterCity +1; % How many data points were added
                    if j == 3
                    fillMinterCity_prcp = fillMinterCity_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end 
end

% Determine number of no data remaining after aggregation.
noDat_Sunflower = sum(climateOut_Sunflower(:,2)== -9999);
tmp = climateOut_Sunflower(climateOut_Sunflower(:,1) >= 20000101,:);
noDat_Sunflower_subset = sum(tmp(:,2)==-9999);

perc_prcp_Sunflower = (fillSunflower_prcp/total_num_days)*100;
perc_prcp_SandyRidge = (fillSandyRidge_prcp/total_num_days)*100;
perc_prcp_Moorhead = (fillMoorhead_prcp/total_num_days)*100;
perc_prcp_Indianola = (fillIndianola_prcp/total_num_days)*100;
perc_prcp_Shaw01 = (fillShaw01_prcp/total_num_days)*100;
perc_prcp_Cleveland = (fillCleveland_prcp/total_num_days)*100;
perc_prcp_BeasleyLake = (fillBeasleyLake_prcp/total_num_days)*100;
perc_prcp_Cleveland3 = (fillCleveland3_prcp/total_num_days)*100;
perc_prcp_MinterCity = (fillMinterCity_prcp/total_num_days)*100;

% Create stats table
stats_Sunflower = table(categorical({'Sunflower 3.9 N';'Sandy Ridge SCAN Site';'Moorhead';'Indianola 1.1 N';'Shaw 0.1 WSW';'Cleveland';'Beasley Lake SCAN Site';'Cleveland 3 N';'Minter City'}),...
    [perc_prcp_Sunflower;perc_prcp_SandyRidge;perc_prcp_Moorhead;perc_prcp_Indianola;perc_prcp_Shaw01;perc_prcp_Cleveland;perc_prcp_BeasleyLake;perc_prcp_Cleveland3;perc_prcp_MinterCity],...
    'VariableNames',{'Station' 'PercentFilled'});

% Export climate data
T = array2table(climateOut_Sunflower,'VariableNames',{'DATE','PRCP','SNWD','SNOW','TMAX','TMIN','TOBS'});
writetable(T,'Output_Sunflower.csv')

% Export stats
writetable(stats_Sunflower,'Stats_Sunflower.csv')

%%

fprintf('FINI \n');
