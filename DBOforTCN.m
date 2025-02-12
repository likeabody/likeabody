function [fMin , bestX, Convergence_curve,bestnet] = DBOforTCN(pop, M,c,d,dim,inputn,outputn,inputn_test,outputps,output_test,numFeatures,outputSize)

P_percent = 0.2;    % The population size of producers accounts for "P_percent" percent of the total population size


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pNum = round( pop *  P_percent );    % The population size of the producers


lb= c.*ones( 1,dim );    % Lower limit/bounds/     a vector
ub= d.*ones( 1,dim );    % Upper limit/bounds/     a vector
%Initialization
for i = 1 : pop

    x( i, : ) = lb + (ub - lb) .* rand( 1, dim );
    [net0(i),fit( i )] = fun( x( i, : )  ,inputn,outputn,inputn_test,outputps,output_test,numFeatures,outputSize);
    
end
pnet = net0;
pFit = fit;
pX = x;
XX=pX;
[ fMin, bestI ] = min( fit );      % fMin denotes the global optimum fitness value
bestX = x( bestI, : );             % bestX denotes the global optimum position corresponding to fMin
bestnet = net0(bestI);
% Start updating the solutions.
for t = 1 : M

    [fmax,B]=max(fit);
    worse= x(B,:);
    r2=rand(1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1 : pNum
        if(r2<0.9)
            a=rand(1,1);
            if (a>0.1)
                a=1;
            else
                a=-1;
            end
            x( i , : ) =  pX(  i , :)+0.3*abs(pX(i , : )-worse)+a*0.1*(XX( i , :)); % Equation (1)
        else
            aaa= randperm(180,1);
            if ( aaa==0 ||aaa==90 ||aaa==180 )
                x(  i , : ) = pX(  i , :);
            end
            theta= aaa*pi/180;
            x(  i , : ) = pX(i,:)+tan(theta).*abs(pX(i , : )-XX( i , :));    % Equation (2)
        end

        x(i,:) = Bounds(x(i,:),lb,ub);
         [net0(i),fit(i)] = fun( x( i, : )  ,inputn,outputn,inputn_test,outputps,output_test,numFeatures,outputSize);
    end
    [ fMMin, bestII ] = min( fit );      % fMin denotes the current optimum fitness value
    bestXX = x( bestII, : );             % bestXX denotes the current optimum position

    R=1-t/M;                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Xnew1 = bestXX.*(1-R);
    Xnew2 = bestXX.*(1+R);                    %%% Equation (3)
    Xnew1= Bounds( Xnew1, lb, ub );
    Xnew2 = Bounds( Xnew2, lb, ub );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Xnew11 = bestX.*(1-R);
    Xnew22 =bestX.*(1+R);                     %%% Equation (5)
    Xnew11= Bounds( Xnew11, lb, ub );
    Xnew22 = Bounds( Xnew22, lb, ub );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = ( pNum + 1 ) :12                  % Equation (4)
        x( i, : )=bestXX+((rand(1,dim)).*(pX(i,:)-Xnew1)+(rand(1,dim)).*(pX(i,:)-Xnew2));
        x(i, : ) = Bounds( x(i, : ), Xnew1, Xnew2 );
        [net0(i),fit(i )] = fun( x( i, : )  ,inputn,outputn,inputn_test,outputps,output_test,numFeatures,outputSize);
    end

    for i = 13: 19                  % Equation (6)


        x( i, : )=pX( i , : )+((randn(1)).*(pX( i , : )-Xnew11)+((rand(1,dim)).*(pX( i , : )-Xnew22)));
        x(i, : ) = Bounds( x(i, : ),lb, ub);
        [net0(i),fit(i)] = fun( x( i, : )  ,inputn,outputn,inputn_test,outputps,output_test,numFeatures,outputSize);

    end

    for j = 20 : pop                 % Equation (7)
        x( j,: )=bestX+randn(1,dim).*((abs(( pX(j,:  )-bestXX)))+(abs(( pX(j,:  )-bestX))))./2;
        x(j, : ) = Bounds( x(j, : ), lb, ub );
        [net0(j),fit(j )] = fun( x( j, : )  ,inputn,outputn,inputn_test,outputps,output_test,numFeatures,outputSize);
    end
    % Update the individual's best fitness vlaue and the global best fitness value
    XX=pX;
    for i = 1 : pop
        if ( fit( i ) < pFit( i ) )
            pFit( i ) = fit( i );
            pX( i, : ) = x( i, : );
            pnet(i) = net0(i);
        end

        if( pFit( i ) < fMin )
            % fMin= pFit( i );
            fMin= pFit( i );
            bestX = pX( i, : );
            %  a(i)=fMin;
            bestnet = pnet(i);

        end
    end

    Convergence_curve(t)=fMin;
    disp(['第',num2str(t),'次寻优的MSE为：',num2str(fMin)])
end

% Application of simple limits/bounds
function s = Bounds( s, Lb, Ub)
% Apply the lower bound vector
temp = s;
I = temp < Lb;
temp(I) = Lb(I);

% Apply the upper bound vector
J = temp > Ub;
temp(J) = Ub(J);
% Update this new move
s = temp;
function S = Boundss( SS, LLb, UUb)
% Apply the lower bound vector
temp = SS;
I = temp < LLb;
temp(I) = LLb(I);

% Apply the upper bound vector
J = temp > UUb;
temp(J) = UUb(J);
% Update this new move
S = temp;