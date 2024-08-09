### All functions are originally from https://github.com/kassteele/Nowcasting/tree/master and then adapted 
### and extended for the analysis of this paper. Credit to the original creators.

genPriorDelayDist <- function(mean.delay, max.delay, p = 0.999) {
  # Arguments
  # mean.delay  Assumed mean delay in days
  # max.delay   Assumed maximum delay in days, where a fraction p of all cases have been reported
  #             Also the number of days back for which to make a delay correction
  # p           Fraction of reported cases within max.delay days. Default 99.9 %
  #
  # Value
  # PMF as a vector of length max.delay + 1
  # 
  # Details
  # Prior delay distribution is assumed to be Negative Binomial
  # Note that this prior delay distribution is disease specific!
  theta.delay <- exp(uniroot(
    f = function(x) qnbinom(p = p, mu = mean.delay, size = exp(x)) - max.delay,
    interval = c(0, 10), extendInt = "yes")$root)
  
  if(theta.delay <= 0) theta.delay = 0.01
  
  # We expect 1 case on day 1 of the outbreak
  # log(f.priordelay) is then the boundary constraint for the trend surface
  f.priordelay <- 1*dnbinom(x = 0:max.delay, mu = mean.delay, size = theta.delay)
  f.priordelay <- f.priordelay/sum(f.priordelay)
  
  # Return output
  return(f.priordelay)
}

dataSetup <- function(data, start.date, end.date = NULL, nowcast.date, days.back = NULL, f.priordelay) {
  #
  # Data setup
  #
  # Description
  # Function that sets up the data for nowcasting
  #
  # Arguments
  # data          Dateframe with two Date columns: onset.date and report.date
  # start.date    Starting date of outbreak
  # end.date      Ending date of outbreak
  #               In real-time, leave NULL so end.date = nowcast.date
  # nowcast.date  Nowcast date
  # days.back     Number of days back from nowcast.date to include in estimation procedure
  #               If NULL, it is set to two times the number of days in f.priordelay
  # f.priordelay  Prior delay PMF, from genPriorDelayDist
  #
  # Value
  # Dataframe with:
  # Date      date of disease onset
  # Delay     delay (days)
  # Reported  factor with levels: "Reported", "Not yet reported" and, retrospectively, "Future"
  # Day       factor with day of the week
  # Cases     number of cases
  # Est       include record in estimation procedure (1 = yes, 0 = no)
  # t         number of days since start.date
  # d         delay (days)
  # g         boundary constraint, log(reporting intensity)
  # b         boundary constraint indicator (1 = active, 0 = not active)
  
  #
  # Initial stuff
  #
  
  # If there is no end.date, set end.date equal to nowcast.date
  if (is.null(end.date)) end.date <- nowcast.date
  
  # Get maximum delay
  max.delay <- length(f.priordelay) - 1
  
  # If there is no days.back, set days.back to two times max.delay
  if (is.null(days.back)) days.back <- 2*max.delay
  
  # Get the dimensions of the reporting trapezium (= T x D1) and the entire outbreak (= T.true x D1)
  T      <- as.numeric(nowcast.date - start.date) + 1 # Number of days from start.date to nowcast.date
  T.true <- as.numeric(    end.date - start.date) + 1 # Number of days from start.date to end.date (truth, retrospectively)
  D      <- max.delay                                 # Maximum delay in days
  D1     <- D + 1                                     # Number of days from 0 to max.delay
  
  # Set vectors t, t.true and d
  t      <- 1:T      # Days since start.date: 1, 2, ..., T
  t.true <- 1:T.true # Days since start.date: 1, 2, ..., T.true
  d      <- 0:D      # Delays 0, 1, ..., max.delay
  
  #
  # Data operations
  #
  
  data <- data %>%
    # Filter records with start.date <= onset.data <= end.date
    filter(onset.date >= start.date & onset.date <= end.date) %>% 
    
    # Compute delay in days
    mutate(delay = (report.date - onset.date) %>% as.numeric) %>% 
    
    # Filter records with 0 <= delay <= max.delay
    filter(delay >= 0 & delay <= max.delay) %>% 
    
    # Categorize onset.date and delay
    # We need this to tabulate the cases by onset.date and delay
    mutate(
      onset.date.cat = onset.date  %>% cut(breaks = seq(from = start.date, to = end.date + 1, by = "day")),
      delay.cat      = delay %>% factor(levels = 0:max.delay)) %>% 
    
    # Remove (numeric) delay
    select(-delay)
  
  #
  # Construct reporting data
  #
  
  # Setup the reporting trapezium data as a grid by Date and Delay
  rep.data <- expand.grid(
    Date = data$onset.date.cat %>% levels %>% as.Date,
    Delay = d) %>% 
    
    mutate(
      # Add t (t.true, actually) and d, to assist in the calculations
      t = (Date - start.date + 1) %>% as.integer,
      d = Delay,
      
      # Add reporting category: Reported, Not yet reported, Future
      # Cases with t + d <= T have been reported
      Reported = ifelse(t + d <= T, yes = "Reported",
                        # Cases with t > T are in the future
                        no = ifelse(t > T, yes = "Future",
                                    # The rest has not been reported yet
                                    no = "Not yet reported")) %>% 
        # As factor
        factor(levels = c("Reported", "Not yet reported", "Future")),
      
      # Add day of the week
      # [t = 3, d = 0], [t = 2, d = 1], [t = 1, d = 2] = constant, etc., so we have
      # Monday is reference (trick: 2007-01-01 is Monday)
      Day = weekdays(
        x = Date + Delay,
        abbreviate = TRUE) %>%
        factor(
          levels = weekdays(
            x = seq(
              from = as.Date("2007-01-01"),
              to = as.Date("2007-01-07"),
              by = "1 day"),
            abbreviate = TRUE)),
      
      # Add tabulated cases by date and delay
      Cases = with(data, table(onset.date.cat, delay.cat)) %>% as.vector,
      
      # Include record in estimation procedure? (1 = yes, 0 = no)
      # This is where nowcast.date - days.back + 1 <= Onset.Date <= nowcast.date
      Est = (Date >= (nowcast.date - days.back + 1) & Date <= nowcast.date) %>% as.integer)
  
  #
  # Add boundary constraints to reporting data
  #
  
  # g is value to keep surface below eta <= g
  # b is where the constraint is active
  g <- matrix(0, nrow = T.true, ncol = D1)
  b <- matrix(0, nrow = T.true, ncol = D1)
  # Set g and b at t = 1
  g[1, ] <- log(f.priordelay)
  b[1, ] <- 1
  # Set g and b at max.delay
  g[, D1] <- log(f.priordelay[D1])
  b[, D1] <- 1
  # Add these to rep.data
  rep.data <- rep.data %>%
    mutate(
      b = b %>% as.vector,
      g = g %>% as.vector)
  
  #
  # Return output
  #
  
  return(rep.data)
}

plotTrapezoid <- function(data, title = "Reporting trapezoid") {
  #
  # Descrition
  # Plot reporting trapezoid
  #
  # Arguments
  # data   Dataframe, output from dataSetupt
  # title  Title of the plot
  #
  # Value
  # ggplot object
  
  # Make the plot
  plot <- ggplot(
    data = data,
    mapping = aes(x = Date, y = Delay, fill = rescale(log1p(Cases)) + 100*(as.numeric(Reported) - 1))) +
    geom_raster() +
    scale_fill_gradientn(
      limits = c(0, 201),
      colours = c(blue.pal(n = 5), oran.pal(n = 5), grey.pal(n = 5)),
      values = rescale(c(outer(seq(from = 0, to = 1, length = 5), c(0, 100, 200), "+")))) +
    scale_x_date(
      date_breaks = "1 month",
      date_labels = "%b %d",
      expand = c(0, 0)) +
    scale_y_continuous(
      expand = c(0, 0)) +
    coord_fixed(
      ratio = 1) +
    labs(
      x = "Time of symptoms onset",
      y = "Reporting delay",
      title = paste(title, "-", format(with(subset(data, Reported == "Reported"), max(Date)), format = "%b %d, %Y"))) +
    theme_bw() +
    theme(
      plot.margin = margin(t = 0.1, r = 0.5, b = 0.1, l = 0.1, unit = "cm"),
      plot.title = element_text(hjust = 0.5),
      legend.position = "none")
  
  # Return output
  return(plot)
  
}

modelSetup <- function(data, ord = 2, kappa = list(u = 1e6, b = 1e6, w = 0.01, s = 1e-6)) {
  #
  # Create model setup
  #
  # Description
  # Function that sets up the nowcasting model
  #
  # Arguments
  # data  Dataframe, output from dataSetup
  #
  # Value
  # List with:
  # matrices  List of model matrices and penalty matrices
  # kappa     Vector with fixed smoothing parameters for constraints
  
  #
  # Initial stuff
  #
  
  # Filter on records with Est == 1
  data <- data %>% filter(Est == 1)
  
  # Extract dimensions
  t <- unique(data$t)
  d <- unique(data$d)
  T  <- length(t)
  D1 <- length(d)
  
  #
  # Model matrices
  #
  
  # B-spline basis matrix for smooth surface
  Bt <- bbase(x = t, k = max(4, floor( T/5)), deg = 3)
  Bd <- bbase(x = d, k = max(4, floor(D1/5)), deg = 3)
  B <- kronecker(Bd, Bt)
  
  # Model matrix for weekday effect
  # Because intercept is included in B-spline basis, drop first column (Monday = reference) of X
  X <- sparse.model.matrix(~ Day, data = data)[, -1]
  
  # cbind them together
  BX <- cbind(B, X)
  
  # Get number of coefficients
  Kt <- ncol(Bt)
  Kd <- ncol(Bd)
  Kw <- ncol(X)
  
  #
  # Difference operator and penalty matrices
  #
  
  # Difference operator matrices
  Dt <- kronecker(Diagonal(Kd), diff(Diagonal(Kt), diff = ord)) # Smoothness in t direction
  Dd <- kronecker(diff(Diagonal(Kd), diff = 2), Diagonal(Kt))   # Smoothness in d direction
  Du <- kronecker(diff(Diagonal(Kd), diff = 2), Diagonal(Kt))   # Unimodal in d direction
  
  # Penalty matrices
  Pt <- t(Dt) %*% Dt
  Pd <- t(Dd) %*% Dd
  Pw <- Diagonal(Kw)
  Ps <- Diagonal(Kt*Kd)
  
  #
  # Fixed smoothing parameters
  #
  
  # kappa.u and kappa.b are large for asymmetric penalty
  # kappa.s is very small for ridge penalty on surface
  # kappa.w is small for ridge penalty on weekday effect
  kappa.u <- kappa$u
  kappa.b <- kappa$b
  kappa.w <- kappa$w
  kappa.s <- kappa$s
  
  #
  # Return output
  #
  
  return(list(
    matrices = list(B = B, X = X, BX = BX, Pt = Pt, Pd = Pd, Du = Du, Ps = Ps, Pw = Pw),
    kappa = c(kappa.u = kappa.u, kappa.b = kappa.b, kappa.w = kappa.w, kappa.s = kappa.s)))
  
}

calculate_bounds <- function(level) {
  lower_bound <- (1 - level) / 2
  upper_bound <- (1 + level) / 2
  if(lower_bound == upper_bound) return(lower_bound)
  else return(c(lower_bound, upper_bound))
}

nowcast <- function(data, model, levels = c(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)) {
  #
  # Nowcasting
  #
  # Description
  # Perform nowcast on data using constraind P-spline smoothing
  #
  # Arguments
  # data        Dataframe with data generated by dataSetup
  # model       List with model setup generated by modelSetup
  # conf.level  Confidence level of the prediction interval. Default 90 %
  #
  # Value
  # List with:
  # nowcast    Dataframe with nowcast statistics (med, lwr, upr) by date 1:T
  # F.nowcast  List of length emprical predictive distribution functions by date 1:T
  # f.delay    Dataframe with delay distribution (PMF) by date 1:T and delay 0:D
  
  #
  # Initial stuff
  #
  
  # Filter on records with Est == 1
  data <- data %>% filter(Est == 1)
  
  # Extract data
  n <- data$Cases
  r <- 2 - as.numeric(data$Reported)
  
  # Extract dimensions
  T  <- data$t %>% unique %>% length
  D1 <- data$d %>% unique %>% length
  
  # Extract matrices
  B  <- model$matrices$B
  X  <- model$matrices$X
  BX <- model$matrices$BX
  
  # Get number of coefficients
  Ks <- ncol(B)
  Kw <- ncol(X)
  
  #
  # Estimate parameters
  #
  
  # Initial alpha, beta and theta
  alpha.beta0 <- coef(lm(log(data$Cases + 0.1) ~ as.matrix(BX) - 1))
  theta0 <- 2
  
  # Estimate parameters
  opt <- greedyGridSearch(
    # Function to be optimized
    fn = function(lambda, ...) estimateAlphaBetaTheta(lambda = lambda, ...)$bic,
    # Set lower and upper boundaries for lambda's
    start = c(10, 1e-4),
    lower = c(10, 1e-4)/100,
    upper = c(10, 1e-4)*100,
    # Set grid size
    n.grid = 21,
    # Optimize lambda's on log-scale
    log = c(TRUE, TRUE),
    # Pass data, model and starting values to fn
    data = data,
    model = model,
    alpha.beta = alpha.beta0,
    theta = theta0)
  
  # Get final parameter estimates after optimization of lambda's
  fit <- estimateAlphaBetaTheta(
    lambda = opt$par,
    data = data,
    model = model,
    alpha.beta = alpha.beta0,
    theta = theta0)
  alpha.beta     <- fit$alpha.beta
  alpha.beta.cov <- fit$alpha.beta.cov
  theta          <- fit$theta
  alpha <- alpha.beta[1:Ks]
  beta  <- alpha.beta[(Ks + 1):(Ks + Kw)]
  
  #
  # Nowcast
  #
  
  # 1. Generate n.samples of the parameter estimates
  #    alpha.beta.sim is a Ks + Kw x n.samples matrix
  n.samples <- 1000
  alpha.beta.sim <- alpha.beta +
    (alpha.beta.cov %>% chol %>% t)%*%matrix(
      rnorm(n = (Ks + Kw)*n.samples),
      nrow = Ks + Kw,
      ncol = n.samples)
  
  # 2. Generate n.sim realizations for the not-yet-reported eta and mu
  #    eta.sim and mu.sim are sum(!r) x n.samples matrices
  eta.sim <- as.matrix(BX[!r, ] %*% alpha.beta.sim)
  mu.sim  <- exp(eta.sim)
  
  # 3. Generate n.samples for the not-yet-reported cases
  #    The already reported cases n are fixed!
  #    n.sim is an T x D1 x n.samples array
  n.sim <- array(n, dim = c(T, D1, n.samples))
  for (i in 1:n.samples) {
    n.sim[, , i][!r] <- rnbinom(
      n = sum(!r),
      mu = mu.sim[, i],
      size = theta)
  }
  
  # 4. Sum over delays by date (keep margins 1 and 3) = epicurve
  #    N.sim is a T x n.samples matrix
  N.sim <- apply(
    X = n.sim,
    MARGIN = c(1, 3),
    FUN = sum)
  
  # 5. Get empirical cumulative predictive distribution function by date (keep margin 1)
  #    F.N is a list of length T with ECDFs
  F.N <- apply(
    X = N.sim,
    MARGIN = 1,
    FUN = ecdf)
  # Additionally, calculate statistics from F.N
  N.stat <- t(sapply(
    X = F.N,
    FUN = quantile,
    probs = unlist(lapply(levels, calculate_bounds)))) # müsste oben unten für alle machen
  colnames(N.stat) <- unlist(lapply(levels, calculate_bounds))
  
  #
  # Delay distribution (PMF) by date
  #
  
  # Surface is for Monday, but is the same for any other day because of division by row sums
  # f.delay is a T x D1 matrix
  eta.s <- B %*% alpha %>% as.vector %>% matrix(nrow = T, ncol = D1)
  mu.s <- exp(eta.s)
  f.delay <- mu.s/rowSums(mu.s)
  
  #
  # Return output
  #
  
  return(list(
    # Nowcast statistics by date
    nowcast = cbind(
      data.frame(Date = data$Date %>% unique),
      as.data.frame(N.stat)),
    # Nowcast predictive distributions (CDF) by date
    F.nowcast = F.N, 
    # Delay distribution (PMF) by date
    f.delay = cbind(
      data[, c("Date", "Delay", "Reported")],
      data.frame(f.delay = as.vector(f.delay)))))
  
}

plotEpicurve <- function(data, title = "Epicurve") {
  #
  # Descrition
  # Plot epidemic curve of delayed data
  #
  # Arguments
  # data    Dataframe generated by dataSetup
  # title   Title of the plot
  #
  # Value
  # ggplot object
  
  # Prepare data for epicurve plot
  tmp <- data %>%
    group_by(Date, Reported) %>%
    summarize(Cases = sum(Cases))
  
  # Make the plot
  plot <- ggplot(
    data = tmp,
    mapping = aes(x = Date, y = Cases, fill = Reported)) +
    geom_col(
      width = 1,
      position = position_stack(reverse = TRUE)) +
    scale_fill_manual(
      values = c(blue, oran, grey),
      name = "") +
    scale_x_date(
      date_breaks = "1 month",
      date_labels = "%b %d",
      expand = c(0, 0)) +
    scale_y_continuous(
      limits = c(0, with(data, max(tapply(Cases, Date, sum)) + 1)),
      expand = c(0, 0)) +
    labs(
      x = "Time of symptoms onset",
      y = "Number of symptomatic cases",
      title = paste(title, "-", format(with(subset(data, Reported == "Reported"), max(Date)), format = "%b %d, %Y"))) +
    theme_bw() +
    theme(
      plot.margin = margin(t = 0.1, r = 0.5, b = 0.1, l = 0.1, unit = "cm"),
      plot.title = element_text(hjust = 0.5),
      legend.position = "top")
  
  # Return output
  return(plot)
  
}

plotTrapezoid <- function(data, title = "Reporting trapezoid") {
  #
  # Descrition
  # Plot reporting trapezoid
  #
  # Arguments
  # data   Dataframe, output from dataSetupt
  # title  Title of the plot
  #
  # Value
  # ggplot object
  
  # Make the plot
  plot <- ggplot(
    data = data,
    mapping = aes(x = Date, y = Delay, fill = rescale(log1p(Cases)) + 100*(as.numeric(Reported) - 1))) +
    geom_raster() +
    scale_fill_gradientn(
      limits = c(0, 201),
      colours = c(blue.pal(n = 5), oran.pal(n = 5), grey.pal(n = 5)),
      values = rescale(c(outer(seq(from = 0, to = 1, length = 5), c(0, 100, 200), "+")))) +
    scale_x_date(
      date_breaks = "1 month",
      date_labels = "%b %d",
      expand = c(0, 0)) +
    scale_y_continuous(
      expand = c(0, 0)) +
    coord_fixed(
      ratio = 1) +
    labs(
      x = "Time of symptoms onset",
      y = "Reporting delay",
      title = paste(title, "-", format(with(subset(data, Reported == "Reported"), max(Date)), format = "%b %d, %Y"))) +
    theme_bw() +
    theme(
      plot.margin = margin(t = 0.1, r = 0.5, b = 0.1, l = 0.1, unit = "cm"),
      plot.title = element_text(hjust = 0.5),
      legend.position = "none")
  
  # Return output
  return(plot)
  
}

bbase <- function(x, x.min = min(x), x.max = max(x), k = 15, deg = 3, sparse = TRUE) {
  #
  # bbase
  #
  # Description
  # Generates design matrix for B-splines
  #
  # Arguments
  # x       A numeric vector of values at which to evaluate the B-spline functions 
  # x.min   Lowest value, min(x)
  # x.max   Highest value, max(x)
  # k       Number of B-spline basis functions
  # deg     Degree of the B-spline basis function. Default is cubic B-splines
  # sparse  Logical indicating if the result should inherit from class "sparseMatrix" (from package Matrix)
  #
  # Value
  # Matrix B-spline basis functions
  
  dx <- (x.max - x.min)/(k - deg)
  knots <- seq(from = x.min - deg*dx, to = x.max + deg*dx, by = dx)
  B <- splines::splineDesign(x = x, knots = knots, ord = deg + 1, outer.ok = TRUE, sparse = sparse)
  return(B)
}

greedyGridSearch <- function(fn, lower, upper, n.grid, start, logscale, ...) {
  # 
  # Optimization over a parameter grid
  # 
  # Description
  # This function does a greedy grid search in any dimension
  #
  # Arguments
  # fn	      A function to be minimized, with first argument the vector of parameters
  #           over which minimization is to take place. It should return a scalar result
  # lower	    Numeric vector containing the lower bounds on the parameter grid
  # upper     Numeric vector containing the upper bounds on the parameter grid
  # n.grid    Integer number determining grid length in every dimension
  # start     Optional numeric vector containing initial values for the parameters to be optimized over
  # logscale  Logical vector. If TRUE, a logarithmic scale is used for that parameter. It defaults to FALSE, i.e., a linear scale
  # ...       Further arguments to be passed to fn
  #
  # Value
  # A list with components:
  # par	      The best set of parameters found
  # value	    The value of fn corresponding to par
  # counts    A integer giving the number of calls to fn
  #
  # Details
  # A greedy algorithm is an algorithmic paradigm that follows the problem solving heuristic of making the
  # locally optimal choice at each stage with the hope of finding a global optimum. In many problems,
  # a greedy strategy works well if there are no local optima.
  
  # Get dimension of parameter space
  n.par <- length(lower)
  # Apply log-transformation to elements of lower and upper?
  lower <- ifelse(test = logscale, yes = log10(lower), no = lower)
  upper <- ifelse(test = logscale, yes = log10(upper), no = upper)
  # Set all possibles values on parameter grid. Result is n.grid x n.par matrix
  par.grid <- mapply(FUN = seq, from = lower, to = upper, length = n.grid)
  # Get initial grid index value
  if (missing(start)) {
    # If start is not given, take index half way the grid
    index <- rep(floor(n.grid/2), times = n.par)
  } else {
    # Else, take index of par.grid value closest to start
    start <- ifelse(test = logscale, yes = log10(start), no = start)
    index <- apply(X = abs(t(par.grid) - start), MARGIN = 1, FUN = which.min)
  }
  # Apply backtransformation to vectors of par.grid?
  par.grid <- matrix(
    mapply(FUN = function(x, logscale) ifelse(test = logscale, yes = 10^x, no = x), t(par.grid), logscale),
    ncol = n.par, byrow = TRUE)
  # Set initial parameter values
  par <- diag(par.grid[index, ])
  # Create n.par dimensional array of size rep(n.grid, n.par) filled with NA
  # Will be filled with function evaluations
  f.eval <- array(NA, dim = rep(n.grid, times = n.par))
  # Initial function value
  f.min <- Inf
  # Initital number of function evaluations
  n.eval <- 0
  # Set move to TRUE to enable search
  move <- TRUE
  while (move) {
    # Stop searching when no improvement
    move <- FALSE
    # For each parameter
    for (i in 1:n.par) {
      # Set candidate index vector to current index vector
      index.can <- index
      # Move i-th index one down and up
      for (j in max(index[i] - 1, 1):min(index[i] + 1, n.grid)) {
        # Set i-th index of index.can to j
        index.can[i] <- j
        # If there is no function value yet
        if (is.na(f.eval[t(index.can)])) {
          # Copy current parameters to candidates
          par.can <- par
          # Replace i-th candidate parameter by par.grid[j, i]
          par.can[i] <- par.grid[j, i]
          # Evaluate function for par.can
          f.new <- fn(par.can, ...)
          # Increase number of function evaluations by one
          n.eval <- n.eval + 1
          # Replace NA in f.eval by f.new
          f.eval[t(index.can)] <- f.new
        } else {
          # If there was already a function value, set f.new to that value
          f.new <- f.eval[t(index.can)]
        }
        # If f.new is smaller than f.min
        if (is.na(f.new) || !is.finite(f.new)) {
          move <- FALSE
        }
        else if (f.new < f.min) {
          # Update f.min by f.new
          f.min <- f.new
          # Update index
          index[i] <- j
          # Update current parameter values
          par <- par.can
          # Set move = TRUE to continue searching
          move <- TRUE
        }
      }
    }
  }
  # Return output
  return(list(par = par, value = f.min, counts = n.eval))
}

# Estimate alpha, beta and theta given lambda
estimateAlphaBetaTheta <- function(lambda, data, model, alpha.beta, theta) {
  
  #
  # Initial stuff
  #
  
  if(theta < 0) theta = 0.1
  epsilon = 1e-4
  
  # Extract data
  y <- data$Cases
  r <- 2 - as.numeric(data$Reported)
  b <- data$b
  g <- data$g
  
  # Extract matrices
  B <- model$matrices$B
  X <- model$matrices$X
  BX <- model$matrices$BX
  Pt <- model$matrices$Pt
  Pd <- model$matrices$Pd
  Du <- model$matrices$Du
  Ps <- model$matrices$Ps
  Pw <- model$matrices$Pw
  
  # Extract fixed smoothing parameters
  kappa.u <- model$kappa["kappa.u"]
  kappa.b <- model$kappa["kappa.b"]
  kappa.w <- model$kappa["kappa.w"]
  kappa.s <- model$kappa["kappa.s"]
  
  # Get number of coefficients
  Ks <- ncol(B)
  Kw <- ncol(X)
  
  #
  # Estimate parameters
  #
  
  # Initial log-likelihood for the negative binomial distribution
  ll <- 10; ll.old <- 1
  it <- 1
  # Penalized iterative weighted least squares algorithm
  while (abs((ll - ll.old)/ll.old) > 1e-2 & it <= 50) {
    # Get alpha and beta
    alpha <- alpha.beta[1:Ks]
    beta  <- alpha.beta[(Ks + 1):(Ks + Kw)]
    # Linear predictor
    eta.s <- as.vector(B %*% alpha)
    eta.w <- as.vector(X %*% beta)
    eta <- eta.s + eta.w
    # Inverse link function: exp(eta)
    mu <- exp(eta)
    # Update theta - for some values goes to 0 so ensure doesn't happen
    terminate_early <- FALSE  # Global flag for early termination
    
    # Custom objective function with early stopping
    objective_function <- function(log.theta) {
      # Transform log.theta back to theta
      theta_val <- exp(log.theta)
      
      # Early return if theta drops below epsilon
      if (theta_val < epsilon) {
        cat("Theta value dropped below epsilon. Stopping optimization.\n")
        terminate_early <<- TRUE  # Set the global flag to indicate termination
        return(Inf)  # Return Inf to stop optimization
      }
      
      # Calculate the negative log-likelihood
      negative_log_likelihood <- -sum(r * dnbinom(x = y, mu = mu, size = theta_val, log = TRUE))
      
      # Debugging outputs to trace values
      #cat("theta:", theta_val, " log(theta):", log.theta, " negative log-likelihood:", negative_log_likelihood, "\n")
      
      # Ensure the result is finite
      if (!is.finite(negative_log_likelihood)) {
        cat("Encountered non-finite negative log-likelihood.\n")
        return(Inf)
      }
      
      return(negative_log_likelihood)
    }
    
    # Run the optimization within tryCatch to handle potential errors
    result <- tryCatch({
      opt.theta <- optim(
        par = log(theta),
        fn = objective_function,
        method = "L-BFGS-B",
        lower = log(epsilon),  # Lower bound for log(theta)
        upper = Inf,           # No upper bound
        control = list(factr = 1e-2),
        hessian = TRUE
      )
      list(success = TRUE, opt.theta = opt.theta)
    }, error = function(e) {
      cat("Optimization failed: ", e$message, "\n")
      list(success = FALSE, opt.theta = NULL)
    })
    
    # Check the result and set theta accordingly
    if (result$success && !terminate_early) {
      theta <- exp(result$opt.theta$par)  # Use the optimized value
    } else {
      theta <- epsilon  # Set theta to epsilon if optimization failed or early termination
    }
    
    # Weights: W = [1/Var(y)]*dmu.deta^2
    W <- (1/(mu + mu^2/theta))*mu^2
    # Working variable: z = eta + (y - mu)*(1/dmu.deta)
    z <- eta + (y - mu)*(1/mu)
    # Unimodal constraint
    vu <- as.numeric(Du %*% alpha >= 0)
    Vu <- Diagonal(x = vu)
    Pu <- t(Du) %*% Vu %*% Du
    # Boundary constraint
    vb <- b*(eta.s >= g)
    Vb <- Diagonal(x = vb)
    XVbX <- t(B) %*% Vb %*% B
    XVbg <- t(B) %*% Vb %*% g
    # Penalty matrix
    P <- bdiag(lambda[1]*Pt + lambda[2]*Pd + kappa.u*Pu + kappa.b*XVbX + kappa.s*Ps, kappa.w*Pw)
    # Normal equations for weighted least squares
    XWX <- t(BX) %*% Diagonal(x = W*r) %*% BX
    XWz <- t(BX) %*% (W*r*z)
    #XWX.P.inv <- solve(XWX + P)
    XWX.P.inv <- chol2inv(chol(XWX + P))
    # Update alpha and beta
    alpha.beta <- XWX.P.inv %*% (XWz + c(kappa.b*as.vector(XVbg), rep(0, Kw)))
    # Calculate log-likelihood for given theta
    ll.old <- ll
    ll <- sum(r*dnbinom(x = y, mu = mu, size = theta, log = TRUE))
    if (!is.finite(ll) || is.na(ll)) {
      ll <- ll.old
      it <- 51 # stop loop
      break
    }
    # Update iterator
    it <- it + 1
  }
  
  #
  # Calculate information criterion
  #
  
  # Calculate effective dimension
  ed <- sum(diag(XWX.P.inv %*% XWX))
  # Calculate BIC
  bic <- -2*ll + log(sum(r))*ed
  
  #
  # Return output
  #
  
  return(list(
    alpha.beta = alpha.beta, alpha.beta.cov = XWX.P.inv,
    theta = theta,
    bic = bic))
}