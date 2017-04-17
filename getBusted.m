function [A] = getBusted(feat, degree,n)
    A(1:n-1,1:feat*degree-1)=0;
    for k=0 : degree
        A(1:n-1, k*feat+1:k*feat+feat)=k;
    end;
end
