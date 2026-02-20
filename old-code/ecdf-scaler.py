class ECDFScaler(Scaler):
    def scaling_fn(self, x):
        _fn = lambda x, *args: npdist.Normal().icdf(self._ecdf_1d(x, *args))
        return jit(vmap(_fn, in_axes=(1, 1, 1, 1), out_axes=1))(x, self.sorted_data, self.ecdf_values, self.slope_at_endpoints)

    def scaling_fn_inv(self, y):
        _fn = lambda y, *args: self._ecdf_inverse_1d(npdist.Normal().cdf(y), *args)
        return jit(vmap(_fn, in_axes=(1, 1, 1, 1), out_axes=1))(y, self.sorted_data, self.ecdf_values, self.slope_at_endpoints)
    
    # def _interp_ecdf_1d(self, x, sorted_col, ecdf_col):
    #     return ipx.interp1d(xq=x, x=sorted_col, f=ecdf_col, method='linear')
    
    # def _interp_ecdf_inverse_1d(self, y, sorted_col, ecdf_col):
    #     fn = lambda x, args: self._interp_ecdf_1d(x, sorted_col, ecdf_col) - y
    #     solver = optx.Bisection(rtol=1e-3, atol=1e-3, flip=False)
    #     # Find the index closest to y without going past it
    #     upper_index = jnp.argmin(y > ecdf_col)
    #     options = {
    #         'lower': sorted_col[upper_index - 1],
    #         'upper': sorted_col[upper_index]
    #     }
    #     # x_guess = ipx.interp1d(xq=y, x=ecdf_col, f=sorted_col, method='linear')
    #     # jax.debug.print("options: for input {input}, {x} < {y} < {z}", input=y, x=options['lower'], y=x_guess, z=options['upper'])
    #     x0 = (sorted_col[upper_index - 1] + sorted_col[upper_index])/2
    #     x = jnp.piecewise(y, [y < ecdf_col[0], y > ecdf_col[-1]], [lambda _: options['lower'], lambda _: options['upper'], lambda _: optx.root_find(fn, solver, x0, options=options, throw=False).value])
    #     return x
    
    # Same as above, but make the root finding happen on the forward implementation
    def _interp_ecdf_1d(self, x, sorted_col, ecdf_col):
        fn = lambda y, args: self._interp_ecdf_inverse_1d(y, sorted_col, ecdf_col) - x
        solver = optx.Bisection(rtol=1e-5, atol=1e-5, flip=False)
        # Find the index closest to y without going past it
        upper_index = jnp.argmin(x > sorted_col)
        options = {
            'lower': ecdf_col[upper_index - 1],
            'upper': ecdf_col[upper_index]
        }
        # y_guess = ipx.interp1d(xq=y, x=sorted_col, f=ecdf_col, method='linear')
        # jax.debug.print("options: for input {input}, {x} < {y} < {z}", input=x, x=options['lower'], y=y_guess, z=options['upper'])
        y0 = (ecdf_col[upper_index - 1] + ecdf_col[upper_index])/2
        y = jnp.piecewise(x, [x < sorted_col[0], x > sorted_col[-1]], [lambda _: options['lower'], lambda _: options['upper'], lambda _: optx.root_find(fn, solver, y0, options=options, max_steps=64, throw=False).value])
        return y

    # TODO: ggrad of interp
    
    def _interp_ecdf_inverse_1d(self, y, sorted_col, ecdf_col):
        return ipx.interp1d(xq=y, x=ecdf_col, f=sorted_col, method='akima')

    def _ecdf_1d(self, x, sorted_col, ecdf_col, slope_at_endpoints):
        middle_ecdf = lambda x, sorted_col, ecdf_col, slope_at_endpoints: self._interp_ecdf_1d(x, sorted_col, ecdf_col)
        left_ecdf = lambda x, sorted_col, ecdf_col, slope_at_endpoints: shifted_scaled_tanh(x, x_shift=sorted_col[0], y_shift=ecdf_col[0], y_scale=ecdf_col[0], x_scale=slope_at_endpoints[0]/ecdf_col[0])
        right_ecdf = lambda x, sorted_col, ecdf_col, slope_at_endpoints: shifted_scaled_tanh(x, x_shift=sorted_col[-1], y_shift=ecdf_col[-1], y_scale=1 - ecdf_col[-1], x_scale=slope_at_endpoints[1]/(1 - ecdf_col[-1]))
        return jnp.piecewise(x, [x < sorted_col[0], x > sorted_col[-1]], [left_ecdf, right_ecdf, middle_ecdf], sorted_col, ecdf_col, slope_at_endpoints)
    
    def _ecdf_inverse_1d(self, x, sorted_col, ecdf_col, slope_at_endpoints):
        middle_ecdf_inverse = lambda x, sorted_col, ecdf_col, slope_at_endpoints: self._interp_ecdf_inverse_1d(x, sorted_col, ecdf_col)
        left_ecdf_inverse = lambda x, sorted_col, ecdf_col, slope_at_endpoints: shifted_scaled_arctanh(x, x_shift=ecdf_col[0], y_shift=sorted_col[0], y_scale=ecdf_col[0]/slope_at_endpoints[0], x_scale=1/ecdf_col[0])
        right_ecdf_inverse = lambda x, sorted_col, ecdf_col, slope_at_endpoints: shifted_scaled_arctanh(x, x_shift=ecdf_col[-1], y_shift=sorted_col[-1], y_scale=(1 - ecdf_col[-1])/slope_at_endpoints[1], x_scale=1/(1 - ecdf_col[-1]))
        return jnp.piecewise(x, [x < ecdf_col[0], x > ecdf_col[-1]], [left_ecdf_inverse, right_ecdf_inverse, middle_ecdf_inverse], sorted_col, ecdf_col, slope_at_endpoints)

    def get_params_from_data(self, data: Float[Array, "N d"]):
        def differentiable_ecdf(samples):
            """
            Computes ECDF at each sample point in a way that allows gradients.
            """
            n = samples.shape[0]
            sorted_samples = jnp.sort(samples)
            # Assign each sorted sample a value 1/n
            # This generates a step function
            ecdf_values = jnp.arange(1, n + 1) / n
            return sorted_samples, ecdf_values

        # Sort the data and compute empirical CDF values
        sorted_data, ecdf_values = vmap(differentiable_ecdf, in_axes=1, out_axes=1)(data)
        
        # Thin the data to reduce the number of points
        max_points = 100
        step = lax.cond(sorted_data.shape[0] > max_points, lambda: sorted_data.shape[0]//max_points, lambda: 1)
        thinned_sorted_data = sorted_data[::step]
        raw_thinned_ecdf_values = ecdf_values[::step]
        
        # Train a GP to smooth the empirical CDF values
        def get_smoothed_ecdf_values(sorted_data, raw_ecdf_values, init_gp_params):
            _build_gp = partial(build_gp, meas_noise=0.05)
            params, losses = train_gp_adam(_build_gp, init_gp_params, sorted_data, raw_ecdf_values, 1000, 1e-3)
            gp = _build_gp(params, sorted_data)
            _, cond_gp = gp.condition(raw_ecdf_values, sorted_data)
            return cond_gp.mean
        
        init_params = {
            'log_amplitude': jnp.zeros(sorted_data.shape[1]),
            'log_lengthscales': jnp.zeros(sorted_data.shape[1])
        }
        thinned_ecdf_values = vmap(get_smoothed_ecdf_values, in_axes=(1, 1, 0), out_axes=1)(thinned_sorted_data, raw_thinned_ecdf_values, init_params)
        
        # Remove out of bounds values
        def make_out_of_bounds_mask(ecdf_values):
            below_zero_mask = ecdf_values < 0.0
            above_one_mask = ecdf_values > 1.0
            return below_zero_mask | above_one_mask

        out_of_bounds_masks = vmap(make_out_of_bounds_mask, in_axes=1, out_axes=1)(thinned_ecdf_values)
        thinned_sorted_data = jnp.stack([arr[~mask] for arr, mask in zip(thinned_sorted_data.T, out_of_bounds_masks.T)], axis=1)
        thinned_ecdf_values = jnp.stack([arr[~mask] for arr, mask in zip(thinned_ecdf_values.T, out_of_bounds_masks.T)], axis=1)
        
        # Remove knot points (if needed) to ensure monotonicity
        monotonic_ecdf_mask = vmap(make_monotonic_increasing_mask, in_axes=1, out_axes=1)(thinned_ecdf_values)
        self._sorted_data = jnp.stack([arr[mask] for arr, mask in zip(thinned_sorted_data.T, monotonic_ecdf_mask.T)], axis=1)
        self._ecdf_values = jnp.stack([arr[mask] for arr, mask in zip(thinned_ecdf_values.T, monotonic_ecdf_mask.T)], axis=1)
        
        # Get the slope at the endpoints
        get_slope = lambda x: vmap(grad(lambda x, sorted_col, ecdf_col: self._interp_ecdf_1d(x, sorted_col, ecdf_col)), in_axes=(0, 1, 1), out_axes=0)(x, self._sorted_data, self._ecdf_values)
        self._slope_at_endpoints = jnp.stack([get_slope(self._sorted_data[0]), get_slope(self._sorted_data[-1])], axis=0)
    
    @property
    def sorted_data(self):
        return self._sorted_data
    
    @property
    def ecdf_values(self):
        return self._ecdf_values
    
    @property
    def slope_at_endpoints(self):
        return self._slope_at_endpoints
    
    @sorted_data.setter
    def sorted_data(self, value):
        self._sorted_data = value
    
    @ecdf_values.setter
    def ecdf_values(self, value):
        self._ecdf_values = value
    
    @slope_at_endpoints.setter
    def slope_at_endpoints(self, value):
        self._slope_at_endpoints = value
    
    @property
    def param_names(self):
        return ["sorted_data", "ecdf_values", "slope_at_endpoints"]

