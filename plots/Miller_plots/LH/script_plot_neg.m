Xline= [0,  20];
Yline= [0,  2];
Zline= [4, 3];

color = 'red';
p = plot3(Xline,Yline,Zline,color);
p.LineWidth = 3;
hold on;
%p1= plot3(Xline,Yline,[0,0],color);
p2= plot3(Xline,[0,0],Zline,color);
p3= plot3([0,0],Yline,Zline,color);
%p1.LineStyle = '--';
p2.LineStyle = '--';
p3.LineStyle = '--';
%p1.Color(4)  = 0.5;
p2.Color(4)  = 0.5;
p3.Color(4)  = 0.5;
%p1.LineWidth = 2;
p2.LineWidth = 2;
p3.LineWidth = 2;
hold off;

alpha0=0.5;


name = 'isocontour_eta=0.0_eps=0.3333333333333333_q=3.0_kappa=1.5_delta=-0.5_dR0dr=-0.5_skappa=0.5_sdelta=-0.5773502691896258.hdf5';

% import stuff
omn_mat     = h5read(name, '/omnv' );
alpha_mat   = h5read(name, '/alphav' );
s_mat       = h5read(name, '/sv' );
AE_mat      = h5read(name, '/AEv' );

AE_mat = log10(AE_mat);

AE_max = 2;
AE_min = -1 ;

AE_space = linspace(AE_min,AE_max,6);

surf0=isosurface(omn_mat,alpha_mat,s_mat,AE_mat,AE_space(1));
p0 = patch(surf0);
isonormals(omn_mat,alpha_mat,s_mat,AE_mat,p0);
set(p0,'FaceColor','[0.2322    0.1855    0.4989]','EdgeColor','none','FaceAlpha',alpha0); % set the color


surf1=isosurface(omn_mat,alpha_mat,s_mat,AE_mat,AE_space(2));
p1 = patch(surf1);
isonormals(omn_mat,alpha_mat,s_mat,AE_mat,p1);
set(p1,'FaceColor','[0.2453    0.6073    0.9973]','EdgeColor','none','FaceAlpha',alpha0); % set the color


surf2=isosurface(omn_mat,alpha_mat,s_mat,AE_mat,AE_space(3));
p2 = patch(surf2);
isonormals(omn_mat,alpha_mat,s_mat,AE_mat,p2);
set(p2,'FaceColor','[0.2738    0.9704    0.5186]','EdgeColor','none','FaceAlpha',alpha0); % set the color


surf3=isosurface(omn_mat,alpha_mat,s_mat,AE_mat,AE_space(4));
p3 = patch(surf3);
isonormals(omn_mat,alpha_mat,s_mat,AE_mat,p3);
set(p3,'FaceColor','[0.8825    0.8663    0.2170]','EdgeColor','none','FaceAlpha',alpha0); % set the color


surf4=isosurface(omn_mat,alpha_mat,s_mat,AE_mat,AE_space(5));
p4 = patch(surf4);
isonormals(omn_mat,alpha_mat,s_mat,AE_mat,p4);
set(p4,'FaceColor','[0.9411    0.3562    0.0705]','EdgeColor','none','FaceAlpha',alpha0); % set the color

daspect([4,2,1]);
xlabel('$\tilde{\omega}_n$','Interpreter','latex');
ylabel('$\alpha$','Interpreter','latex');
zlabel('$s$','Interpreter','latex');
grid on;

view([137 20]);




colormap(turbo);
c = colorbar;
caxis([AE_min AE_max]);
c.Label.String = '$\log \widehat{A}$';
c.Label.Interpreter = 'latex';

txt = '$(b)$: $\delta = -0.5$';
text(15.0,8.0,3.5,txt,'Interpreter','latex')


fontname(gcf,"CMU Serif")

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 6, 2.1]);
exportgraphics(gcf,'LH-plot-negtriangle.png','Resolution',2000)