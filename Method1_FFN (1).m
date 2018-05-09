%First import the dataset "pricedata" from home menu in the toolbar

hi=pricedata(:,5);          %closing price of stocks loaded
b=table2array(hi);          %conversion from table to array

in=b(1:50,1);               
t=b(51:80,1);
for i=5:950
        in=horzcat(in,b(i:i+49,1));     %Forming an input array of stock prices
        t=horzcat(t,b(i+51:i+80,1));    %Forming an output array of stock prices
                                        %Of 947 instances with instance
                                        %containing 50 inputs and 30 outputs                                      
end

net=feedforwardnet(100);                %Building a FFN with 100 hidden nodes
net= train(net,in,t);                   %Training the net passing input,output arrays and net as arguments
view(net)                               %Viewing the network

in2=b(1001:1050,1);                     %Input for prediction
y=net(in2);                             %Predicted Values

figure
v1=1051:1080;
v1=v1';
q=b(1051:1080,1);
plot(v1,y)                 %Plotting the predicted values of testing region

hold on
plot(v1,q)                 %Plotting the actual values of testing region

hold on
v2=1:1050;
v2=v2';
q2=b(1:1050,1);            %Plotting the actual values of training region

plot(v2,q2)

hold on
v3=1:30;
v3=v3';                    %Forming an array for plotting against the predicted values
for m=5:950
    v3=horzcat(v3,v2(m:m+29,1));
end
y2=net(in);
plot(v3,y2)                %Plotting the predicted values of training region

