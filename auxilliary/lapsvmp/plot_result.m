
figure
subplot (1,3,1)
hold on
plot(num_pos_label_list,diag(lapsvm_precision))
plot(num_pos_label_list,diag(lgc_precision))
plot(num_pos_label_list,diag(svm_precision))
hold off
xlabel ('#pos')
title ('Precision')
ylabel ('Precision')
legend('LapSVM', 'LGC', 'SVM', 'Location', 'east')
grid

subplot (1,3,2)
hold on
plot(num_pos_label_list,diag(lapsvm_recall))
plot(num_pos_label_list,diag(lgc_recall))
plot(num_pos_label_list,diag(svm_recall))
hold off
xlabel ('#pos')
title ('Recall')
ylabel ('Recall')
legend('LapSVM', 'LGC', 'SVM', 'Location', 'east')
grid

subplot (1,3,3)
hold on
plot(num_pos_label_list,diag(lapsvm_f))
plot(num_pos_label_list,diag(lgc_f))
plot(num_pos_label_list,diag(svm_f))
hold off
xlabel ('#pos')
title ('F-measure')
ylabel ('F-measure')
legend('LapSVM', 'LGC', 'SVM', 'Location', 'east')
grid
