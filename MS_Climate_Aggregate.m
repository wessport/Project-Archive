%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  AGGREGATING CLIMATE DATA  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read Coahoma_v1.csv
climateCoa = dlmread('Coahoma_v1.csv',',',1,1);

% Read Bolivar_v1.csv
climateBol = dlmread('Bolivar_v1.csv',',',1,1);

% Read Sunflower_v1.csv
climateSun = dlmread('Sunflower_v1.csv',',',1,1);

%subset the data based on the climate stations

% Coahoma
Clark29 = climateCoa(climateCoa(:,1)==1,:); % Clarksdale 2.9 SW
fillClark29 = 0;
fillClark29_prcp = 0;

Clark = climateCoa(climateCoa(:,1)==2,:); %Clarksdale (Majority of data)
fillClark = 0;
fillClark_prcp = 0;


%Bolivar
Cleveland3 = climateBol(climateBol(:,1)==1,:); % Cleveland3 N
fillCleveland3 = 0;
fillCleveland3_prcp = 0;

Cleveland = climateBol(climateBol(:,1)==2,:); % Cleveland
fillCleveland = 0;
fillCleveland_prcp = 0;

Duncan01 = climateBol(climateBol(:,1)==3,:); %Duncan0.1 NNW
fillDuncan01 = 0;
fillDuncan01_prcp = 0;

Duncan = climateBol(climateBol(:,1)==4,:); %Duncan
fillDuncan = 0;
fillDuncan_prcp = 0;

Rosedale = climateBol(climateBol(:,1)==5,:); %Rosedale
fillRose = 0;
fillRose_prcp = 0;

Scott = climateBol(climateBol(:,1)==6,:); %Scott
fillScott = 0;
fillScott_prcp = 0;

Shaw01 = climateBol(climateBol(:,1)==7,:); %Shaw 0.1 WSW
fillShaw01 = 0;
fillShaw01_prcp = 0;

%Sunflower
Indianola = climateSun(climateSun(:,1)==1,:); % Indianola 1.1 N
fillIndianola = 0;
fillIndianola_prcp = 0;

Moorhead = climateSun(climateSun(:,1)==2,:); % Moorhead
fillMoorhead = 0;
fillMoorhead_prcp = 0;

Sunflower = climateSun(climateSun(:,1)==3,:); % Sunflower
fillSunflower = 0;
fillSunflower_prcp = 0;


% Complete list of all dates since 1970 Jan 1
dates = dlmread('Dates.csv',',');

%Create climate output file where aggregated data will go
climateOut = zeros(length(dates),5);

climateOut(:,1) = dates;


%% 
% Loop through each column in climate
% Includes PRCP,SNWD,SNOW,TMAX,TMIN,TOBS

for j = 3:8


    % Populating climate data from Clarksdale (Clark - most complete record)
    station = Clark; 

    %Check to see if station includes data before 1 Jan 1970
    % If it does, exclude that data

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101,:);
        station = R;
    end

    %Initialize for loop
    count = 1;
    index = 1;
 
    % If the station has data, append it for the respective date 
    % to the climateOut array
    for i = 1:length(dates)
        if count < length(station) + 1
            if dates(i,1) == station(count,2)
                climateOut(index,j-1) = station(count,j);
                count = count + 1;
                index = index + 1;
                fillClark = fillClark +1;
                if j == 3
                    fillClark_prcp = fillClark_prcp +1;
                end
            
            else
                climateOut(index,j-1) = -9999;
                index = index + 1;
            end
        else
            climateOut(index,j-1) = -9999;
            index = index + 1;
        end             
    end
    %% 

    % Add Clarksdale 2.9 SW to fill in no data gaps
    station = Clark29;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2016
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut(i,j-1) = station(count,j); % Add that data for that date
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

     %% 
    % Add Cleveland3 N to fill in no data gaps
    station = Cleveland3;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2016
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillCleveland3 = fillCleveland3 +1; % How many data points were added
                    if j == 3
                    fillCleveland3_prcp = fillCleveland3_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end    
    
    %% 
    % Add Cleveland to fill in no data gaps
    station = Cleveland;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2016
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut(i,j-1) = station(count,j); % Add that data for that date
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
    
     %% 
    % Add Duncan0.1 NNW to fill in no data gaps
    station = Duncan01;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2016
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillDuncan01 = fillDuncan01 +1; % How many data points were added
                    if j == 3
                    fillDuncan01_prcp = fillDuncan01_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    
    %% 
    % Add Duncan to fill in no data gaps
    station = Duncan;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2016
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillDuncan = fillDuncan +1; % How many data points were added
                    if j == 3
                    fillDuncan_prcp = fillDuncan_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    %% 
    % Add Rosedale to fill in no data gaps
    station = Rosedale;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2016
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillRose = fillRose +1; % How many data points were added
                    if j == 3
                    fillRose_prcp = fillRose_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    %% 
    % Add Scott to fill in no data gaps
    station = Scott;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2016
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillScott = fillScott +1; % How many data points were added
                    if j == 3
                    fillScott_prcp = fillScott_prcp + 1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    %% 
    % Add Shaw 0.1 to fill in no data gaps
    station = Shaw01;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2016
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut(i,j-1) = station(count,j); % Add that data for that date
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
    
     %% 
    % Add Indianola 1.1 N to fill in no data gaps
    station = Indianola;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2016
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillIndianola = fillIndianola +1; % How many data points were added
                    if j == 3
                    fillIndianola_prcp = fillIndianola_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end
    
    
     %% 
    % Add Moorhead to fill in no data gaps
    station = Moorhead;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2016
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut(i,j-1) = station(count,j); % Add that data for that date
                    count = count + 1; % Move on to the next day value
                    fillMoorhead = fillMoorhead +1; % How many data points were added
                    if j == 3
                    fillMoorhead_prcp = fillMoorhead_prcp +1;
                    end
                else count = count +1;
                end
            end
        end
    end 
    
    
    %% 
    % Add Sunflower to fill in no data gaps
    station = Sunflower;

    if min(station(:,2)) < 19700101
        [R] = station(station(:,2) >= 19700101,:);
        station = R;
    end

    count = 1;
    for i = 1:length(dates) % For all of the dates 1970 - 2016
        if count < length(station) + 1
            if dates(i,1) == station(count,2) % If the dates match
                if station(count,j)~= -9999 && climateOut(i,j-1) == -9999 % And if there's a gap with data to fill it
                    climateOut(i,j-1) = station(count,j); % Add that data for that date
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
end
%% 
% Determine the number of no data remaining after aggregation.
noDat = find(climateOut == -9999);

total_num_days = length(climateOut);
perc_clark_prcp = (fillClark_prcp/total_num_days)*100;
perc_clev_prcp = (fillCleveland_prcp/total_num_days)*100;

% Export climate data
T = array2table(climateOut,'VariableNames',{'DATE','PRCP','SNWD','SNOW','TMAX','TMIN','TOBS'});
writetable(T,'Output_table.csv')

