import torch

def check_convergence(dtheta, j, err_delta, tol_err, tol_delta, max_iters, method='gauss_newton'):
  if torch.norm(dtheta) < tol_delta:
    print('Update got too small at iter %d: %f'%(j,torch.norm(dtheta)))
    return True
  # if torch.norm(err_delta) < tol_err:
    # print('Difference in error got too small at iter %d: %f'%(j ,torch.norm(err_delta)))
    # return True
  if j >= max_iters:
    print('Max iters done')
    return True
  # if method == 'gauss_newton' and err_delta > 0:
  #   print('Error increased')
  #   return True
  return False

def check_convergence_batch(dthetab, j, err_delta, tol_err, tol_delta, max_iters, method='gauss_newton', device=torch.device('cpu')):
  conv_vec = torch.zeros(dthetab.shape[0],1,1).byte()

  dtheta_norm = torch.norm(dthetab.view(dthetab.shape[0], -1), dim=1, p=2)
  err_delta_norm = torch.norm(err_delta.view(dthetab.shape[0], -1), dim=1, p=2)

  conv_vec = torch.where(dtheta_norm < tol_delta, torch.tensor(1,device=device), torch.tensor(0,device=device))#:
    # print('Update got too small at iter %d: %f'%(j,torch.norm(dtheta)))
    # return True
  conv_vec = torch.where(err_delta_norm < tol_err, torch.tensor(1,device=device), torch.tensor(0,device=device))#:
    # print('Difference in error got too small at iter %d: %f'%(j ,torch.norm(err_delta)))
    # return True
  if j >= max_iters:
    print('Max iters done')
    conv_vec = torch.ones(dthetab.shape[0],1,1).byte()
  # if method == 'gauss_newton' and err_delta > 0:
  #   print('Error increased')
  #   return True
  return conv_vec.view(dthetab.shape[0],1,1)

def straight_line_traj(start_conf, goal_conf, traj_time, num_steps, dof, device=torch.device('cpu')):
  th_init = torch.zeros((int(num_steps)+1, 2*dof), device=device)
  avg_vel = (goal_conf - start_conf)/traj_time*1.0
  for i in range(int(num_steps)+1):
    th_init[i, 0:dof] = start_conf*(num_steps - i)*1./num_steps*1. + goal_conf * i*1./num_steps*1. #+ np.array([0., 5.0])
    th_init[i, dof:] = avg_vel

  return th_init

def straight_line_trajb(start_confb, goal_confb, traj_time, num_steps, dof, device=torch.device('cpu')):
  batch_size = start_confb.shape[0]
  th_initb = torch.zeros((batch_size,int(num_steps+1), 2*dof), device=device)
  avg_velb = (goal_confb - start_confb)/traj_time*1.0
  # print th_initb[:,0,0:dof].shape, start_confb[:,0,0:dof].shape, goal_confb[:,0,0:dof].shape
  for i in range(int(num_steps)+1):
    th_initb[:,i,0:dof] = start_confb[:,0,0:dof]*(num_steps-i)*1.0/num_steps*1.0 + goal_confb[:,0,0:dof]*i*1.0/num_steps*1.0
  # print th_initb[:,:,dof:].shape, avg_velb.shape
  th_initb[:,:,dof:]  = avg_velb
  return th_initb



def path_to_traj_avg_vel(path, traj_time, dof, device=torch.device('cpu')):
  num_steps = len(path)
  path = torch.tensor(path)
  start_conf = path[0]
  goal_conf = path[-1]
  th_init = torch.zeros((num_steps, 2*dof), device=device)
  avg_vel = (goal_conf - start_conf)/traj_time*1.0
  for i in range(int(num_steps)):
    th_init[i, 0:dof] = path[i]
    th_init[i, dof:] = avg_vel

  return th_init



def smoothness_metrics(traj, total_time_sec, total_time_step):
  avg_acc = 0.0
  avg_jerk = 0.0
  dtraj = traj[1:,:] - traj[0:-1,:]
  ddtraj = dtraj[1:,:] - dtraj[0:-1,:]
  vel = traj[:, 2:]
  acc = dtraj[:, 2:]/total_time_step*1.0
  jerk = ddtraj[:, 2:]/(total_time_step**2.0)
  vel_magn = torch.norm(vel,p=2,dim=1)
  acc_magn = torch.norm(acc,p=2,dim=1)
  jerk_magn = torch.norm(jerk,p=2,dim=1)
  avg_vel  = torch.mean(vel_magn)
  avg_acc  = torch.mean(acc_magn)
  avg_jerk = torch.mean(jerk_magn)
  
  return avg_vel, avg_acc, avg_jerk

def collision_metrics(traj, obs_error, total_time_sec, total_time_step):
  obs_error = obs_error[1:-1, :] 
  coll_ids = torch.nonzero(obs_error)
  num_penetrating = torch.numel(coll_ids)/2 
  in_coll =  num_penetrating > 0
  avg_penetration = torch.mean(obs_error)
  max_penetration = torch.max(obs_error)
  dt = total_time_sec*1.0/total_time_step*1.0
  coll_intensity = (num_penetrating * dt)/total_time_sec*1.0

  return in_coll, avg_penetration, max_penetration, coll_intensity

