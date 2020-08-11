r"""
Module storing the `~halotools.empirical_models.ModelFactory` class,
an abstract container class used to build
any composite model of the galaxy-halo connection.
"""

import numpy as np
import six
from abc import ABCMeta


from .. import model_defaults

from ...sim_manager import CachedHaloCatalog, FakeSim
from ...sim_manager import sim_defaults
from ...custom_exceptions import HalotoolsError

__all__ = ['ModelFactory']
__author__ = ['Andrew Hearin']


inconsistent_redshift_error_msg = ("Inconsistency between the redshift "
    "already bound to the existing mock = ``%f`` "
    "and the redshift passed as a keyword argument = ``%f``.\n"
    "You should instantiate a new model object if you wish to switch halo catalogs.")
inconsistent_simname_error_msg = ("Inconsistency between the simname "
    "already bound to the existing mock = ``%s`` "
    "and the simname passed as a keyword argument = ``%s``.\n"
    "You should instantiate a new model object if you wish to switch halo catalogs.")
inconsistent_halo_finder_error_msg = ("Inconsistency between the halo-finder "
    "already bound to the existing mock = ``%s`` "
    "and the halo-finder passed as a keyword argument = ``%s``.\n"
    "You should instantiate a new model object if you wish to switch halo catalogs.")
inconsistent_version_name_error_msg = ("Inconsistency between the version_name "
    "already bound to the existing mock = ``%s`` "
    "and the version_name passed as a keyword argument = ``%s``.\n"
    "You should instantiate a new model object if you wish to switch halo catalogs.")


@six.add_metaclass(ABCMeta)
class ModelFactory(object):
    r""" Abstract container class used to build
    any composite model of the galaxy-halo connection.

    See `~halotools.empirical_models.SubhaloModelFactory` for
    subhalo-based models, and
    `~halotools.empirical_models.HodModelFactory` for HOD-style models.
    """

    def __init__(self, input_model_dictionary, **kwargs):
        r"""
        Parameters
        ----------
        input_model_dictionary : dict
            dictionary providing instructions for how to build the composite
            model from a set of components.

        galaxy_selection_func : function object, optional
            Function object that imposes a cut on the mock galaxies.
            Function should take a length-k Astropy table as a single positional argument,
            and return a length-k numpy boolean array that will be
            treated as a mask over the rows of the table. If not None,
            the mask defined by ``galaxy_selection_func`` will be applied to the
            ``galaxy_table`` after the table is generated by the `populate_mock` method.
            Default is None.

        halo_selection_func : function object, optional
            Function object used to place a cut on the input ``table``.
            If the ``halo_selection_func`` keyword argument is passed,
            the input to the function must be a single positional argument storing a
            length-N structured numpy array or Astropy table;
            the function output must be a length-N boolean array that will be used as a mask.
            Halos that are masked will be entirely neglected during mock population.
        """

        # Bind the model-building instructions to the composite model
        self._input_model_dictionary = input_model_dictionary

        try:
            self.galaxy_selection_func = kwargs['galaxy_selection_func']
        except KeyError:
            pass

        try:
            self.halo_selection_func = kwargs['halo_selection_func']
        except KeyError:
            pass

    def populate_mock(self, halocat,
            Num_ptcl_requirement=sim_defaults.Num_ptcl_requirement,
            **kwargs):
        r"""
        Method used to populate a simulation
        with a Monte Carlo realization of a model.

        After calling this method, the model instance
        will have a new ``mock`` attribute.
        You can then access the galaxy population via
        ``model.mock.galaxy_table``, an Astropy `~astropy.table.Table`.

        For documentation specific to the `populate_mock` method of subhalo-based
        models, see `halotools.empirical_models.SubhaloModelFactory.populate_mock`;
        for HOD-style models
        see `halotools.empirical_models.HodModelFactory.populate_mock`.

        See the :ref:`mock_making_tutorials` section of the documentation for
        an in-depth description of the Halotools source-code implementation
        of mock galaxy population.

        Parameters
        ----------
        halocat : object
            Either an instance of `~halotools.sim_manager.CachedHaloCatalog`
            or `~halotools.sim_manager.UserSuppliedHaloCatalog`.

        Num_ptcl_requirement : int, optional
            Requirement on the number of dark matter particles in the halo.
            The column defined by the ``halo_mass_column_key`` string will have a cut placed on it:
            all halos with halocat.halo_table[halo_mass_column_key] < Num_ptcl_requirement*halocat.particle_mass
            will be thrown out immediately after reading the original halo catalog in memory.
            Default value is set in `~halotools.sim_defaults.Num_ptcl_requirement`.
            Currently only supported for instances of `~halotools.empirical_models.HodModelFactory`.

        halo_mass_column_key : string, optional
            This string must be a column of the input halo catalog.
            The column defined by this string will have a cut placed on it:
            all halos with halocat.halo_table[halo_mass_column_key] < Num_ptcl_requirement*halocat.particle_mass
            will be thrown out immediately after reading the original halo catalog in memory.
            Default is 'halo_mvir'.
            Currently only supported for instances of `~halotools.empirical_models.HodModelFactory`.

        masking_function : function, optional
            Function object used to place a mask on the halo table prior to
            calling the mock generating functions. Calling signature of the
            function should be to accept a single positional argument storing
            a table, and returning a boolean numpy array that will be used
            as a fancy indexing mask. All masked halos will be ignored during
            mock population. Default is None.

        enforce_PBC : bool, optional
            If set to True, after galaxy positions are assigned the
            `model_helpers.enforce_periodicity_of_box` will re-map
            satellite galaxies whose positions spilled over the edge
            of the periodic box. Default is True. This variable should only
            ever be set to False when using the ``masking_function`` to
            populate a specific spatial subvolume, as in that case PBCs
            no longer apply.
            Currently only supported for instances of `~halotools.empirical_models.HodModelFactory`.

        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        Notes
        -----
        Note the difference between the
        `halotools.empirical_models.MockFactory.populate` method and the
        closely related method
        `halotools.empirical_models.ModelFactory.populate_mock`.
        The `~halotools.empirical_models.ModelFactory.populate_mock` method
        is bound to a composite model instance and is called the *first* time
        a composite model is used to generate a mock. Calling the
        `~halotools.empirical_models.ModelFactory.populate_mock` method creates
        the `~halotools.empirical_models.MockFactory` instance and binds it to
        composite model. From then on, if you want to *repopulate* a new Universe
        with the same composite model, you should instead call the
        `~halotools.empirical_models.MockFactory.populate` method
        bound to ``model.mock``. The reason for this distinction is that
        calling `~halotools.empirical_models.ModelFactory.populate_mock`
        triggers a large number of relatively expensive pre-processing steps
        and self-consistency checks that need only be carried out once.
        See the Examples section below for an explicit demonstration.

        In particular, if you are running an MCMC type analysis,
        you will choose your halo catalog and completeness cuts, and call
        `~halotools.empirical_models.ModelFactory.populate_mock`
        with the appropriate arguments. Thereafter, you can
        explore parameter space by changing the values stored in the
        ``param_dict`` dictionary attached to the model, and then calling the
        `~halotools.empirical_models.MockFactory.populate` method
        bound to ``model.mock``. Any changes to the ``param_dict`` of the
        model will automatically propagate into the behavior of
        the `~halotools.empirical_models.MockFactory.populate` method.

        Examples
        ----------
        We'll use a pre-built HOD-style model to demonstrate basic usage.
        The same syntax applies to subhalo-based models.

        >>> from halotools.empirical_models import PrebuiltHodModelFactory
        >>> model_instance = PrebuiltHodModelFactory('zheng07')

        Here we will use a fake simulation, but you can populate mocks
        using any instance of `~halotools.sim_manager.CachedHaloCatalog` or
        `~halotools.sim_manager.UserSuppliedHaloCatalog`.

        >>> from halotools.sim_manager import FakeSim
        >>> halocat = FakeSim()
        >>> model_instance.populate_mock(halocat)

        Your ``model_instance`` now has a ``mock`` attribute bound to it.
        You can call the `populate` method bound to the ``mock``,
        which will repopulate the halo catalog with a new Monte Carlo
        realization of the model.

        >>> model_instance.mock.populate()

        If you want to change the behavior of your model, just change the
        values stored in the ``param_dict``. Differences in the parameter values
        will change the behavior of the mock-population.

        >>> model_instance.param_dict['logMmin'] = 12.1
        >>> model_instance.mock.populate()

        See also
        ---------
        :ref:`mock_making_tutorials`

        """
        if hasattr(self, 'redshift'):
            if abs(self.redshift - halocat.redshift) > 0.05:
                raise HalotoolsError("Inconsistency between the model redshift = %.2f"
                    " and the halocat redshift = %.2f" % (self.redshift, halocat.redshift))

        mock_factory_init_args = (
            {'halocat': halocat, 'model': self, 'Num_ptcl_requirement': Num_ptcl_requirement})
        try:
            mock_factory_init_args['halo_mass_column_key'] = kwargs['halo_mass_column_key']
        except KeyError:
            pass
        self.mock = self.mock_factory(**mock_factory_init_args)

        additional_potential_kwargs = ('masking_function', '_testing_mode', 'enforce_PBC', 'seed')
        mockpop_keys = set(additional_potential_kwargs) & set(kwargs)
        mockpop_kwargs = {key: kwargs[key] for key in mockpop_keys}
        self.mock.populate(**mockpop_kwargs)

    def update_param_dict_decorator(self, component_model, func_name):
        r"""
        Decorator used to propagate any possible changes in the composite model param_dict
        down to the appropriate component model param_dict.

        The behavior of the methods bound to the composite model are decorated versions
        of the methods defined in the component models. The decoration is done with
        `update_param_dict_decorator`. For each function that gets bound to the
        composite model, what this decorator does is search the param_dict of the
        component_model associated with the function, and update all matching keys
        in that param_dict with the param_dict of the composite.
        This way, all the user needs to do is make changes to the composite model
        param_dict. Then, when calling any method of the composite model,
        the changed values of the param_dict automatically propagate down
        to the component model before calling upon its behavior.
        This allows the composite_model to control behavior
        of functions that it does not define.

        Parameters
        -----------
        component_model : obj
            Instance of the component model in which the behavior of the function is defined.

        func_name : string
            Name of the method in the component model whose behavior is being decorated.

        Returns
        --------
        decorated_func : function
            Function object whose behavior is identical
            to the behavior of the function in the component model,
            except that the component model param_dict is first updated with any
            possible changes to corresponding parameters in the composite model param_dict.

        See also
        --------
        :ref:`update_param_dict_decorator_mechanism`

        :ref:`param_dict_mechanism`
        """

        # do not pass self into the scope;
        # assuming param_dict is not replaced during life cycle of self.
        # passing `self` causes a cycle reference when
        # the function is assigned as attributes of self.
        __param_dict__ = self.param_dict
        def decorated_func(*args, **kwargs):

            # Update the param_dict as necessary
            for key in list(__param_dict__.keys()):
                if key in component_model.param_dict:
                    component_model.param_dict[key] = __param_dict__[key]

            func = getattr(component_model, func_name)
            return func(*args, **kwargs)

        return decorated_func

    def compute_average_galaxy_clustering(self, num_iterations=5, summary_statistic='median', **kwargs):
        r"""
        Method repeatedly populates a simulation with a mock galaxy catalog, computes the clustering
        signal of each Monte Carlo realization, and returns a summary statistic of the clustering
        such as the median computed from the collection of clustering measurements.

        The `compute_average_galaxy_clustering` is simply a convenience function,
        and is not intended for use in performance-critical applications such as MCMCs.
        In an MCMC, there is no need to repeatedly populate the same snapshot with the same
        set of model parameters; the primary purpose for this repetition is for smoothing out
        numerical noise when making plots and doing exploratory work.
        If you wish to use the 3d correlation function in a performance-critical application,
        see :ref:`galaxy_catalog_analysis_tutorial2` for a demonstration of how to
        call the `~halotools.mock_observables.tpcf` function once,
        directly on the mock galaxy catalog.

        Parameters
        ----------
        num_iterations : int, optional
            Number of Monte Carlo realizations to use to estimate the clustering signal.
            Default is 5.

        summary_statistic : string, optional
            String specifying the method used to estimate the clustering signal from the
            collection of Monte Carlo realizations. Options are ``median`` and ``mean``.
            Default is ``median``.

        simname : string, optional
            Nickname of the simulation into which mock galaxies will be populated.
            Currently supported simulations are
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).
            Default is set in `~halotools.sim_manager.sim_defaults`.

        halo_finder : string, optional
            Nickname of the halo-finder of the halocat into which mock galaxies
            will be populated, e.g., `rockstar` or `bdm`.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        redshift : float, optional
            Redshift of the desired halocat into which mock galaxies will be populated.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        variable_galaxy_mask : scalar, optional
            Any value used to construct a mask to select a sub-population
            of mock galaxies. See examples below.

        mask_function : array, optional
            Function object returning a masking array when operating on the galaxy_table.
            More flexible than the simpler ``variable_galaxy_mask`` option because ``mask_function``
            allows for the possibility of multiple simultaneous cuts. See examples below.

        include_crosscorr : bool, optional
            Only for simultaneous use with a ``variable_galaxy_mask``-determined mask.
            If ``include_crosscorr`` is set to False (the default option), method will return
            the auto-correlation function of the subsample of galaxies determined by
            the input ``variable_galaxy_mask``. If ``include_crosscorr`` is True,
            method will return the auto-correlation of the subsample,
            the cross-correlation of the subsample and the complementary subsample,
            and the the auto-correlation of the complementary subsample, in that order.
            See examples below.

        rbins : array, optional
            Bins in which the correlation function will be calculated.
            Default is set in `~halotools.empirical_models.model_defaults` module.

        Returns
        --------
        rbin_centers : array
            Midpoint of the bins used in the correlation function calculation

        correlation_func : array
            If not using any mask (the default option), method returns the
            correlation function of the full mock galaxy catalog.

            If using a mask, and if ``include_crosscorr`` is False (the default option),
            method returns the correlation function of the subsample of galaxies determined by
            the input mask.

            If using a mask, and if ``include_crosscorr`` is True,
            method will return the auto-correlation of the subsample,
            the cross-correlation of the subsample and the complementary subsample,
            and the the auto-correlation of the complementary subsample, in that order.
            See the example below.

        Examples
        ---------
        The simplest use-case of the `compute_average_galaxy_clustering` function
        is just to call the function with no arguments. This will generate a sequence
        of Monte Carlo realizations of your model into the default halocat,
        calculate the two-point correlation function of all galaxies in your mock,
        and return the median clustering strength in each radial bin:

        >>> model = Leauthaud11() # doctest: +SKIP
        >>> r, clustering = model.compute_average_galaxy_clustering() # doctest: +SKIP

        To control how which simulation is used, you use the same syntax you use to load
        a `~halotools.sim_manager.CachedHaloCatalog` into memory from your cache directory:

        >>> r, clustering = model.compute_average_galaxy_clustering(simname = 'multidark', redshift=1) # doctest: +SKIP

        You can control the number of mock catalogs that are generated via:

        >>> r, clustering = model.compute_average_galaxy_clustering(num_iterations = 10) # doctest: +SKIP

        You may wish to focus on the clustering signal for a specific subpopulation. To do this,
        you have two options. First, you can use the ``variable_galaxy_mask`` mechanism:

        >>> r, clustering = model.compute_average_galaxy_clustering(gal_type = 'centrals') # doctest: +SKIP

        With the ``variable_galaxy_mask`` mechanism, you are free to use any column of your galaxy_table
        as a keyword argument. If you couple this function call with the ``include_crosscorr``
        keyword argument, the function will also return all auto- and cross-correlations of the subset
        and its complement:

        >>> r, cen_cen, cen_sat, sat_sat = model.compute_average_galaxy_clustering(gal_type = 'centrals', include_crosscorr = True) # doctest: +SKIP

        Notes
        -----
        The `compute_average_galaxy_clustering` method
        bound to mock instances is just a convenience wrapper
        around the `~halotools.mock_observables.tpcf` function. If you wish for greater
        control over how your galaxy clustering signal is estimated,
        see the `~halotools.mock_observables.tpcf` documentation.

        Note that there can be no guarantees that the
        `compute_average_galaxy_clustering` method bound to your model
        will terminate in a reasonable amount of time. For example,
        if you use a subhalo-based model that populates *every* subhalo
        in the catalog with a mock galaxy, then calling
        `compute_average_galaxy_clustering` on this model will attempt
        to compute a correlation function on hundreds of millions of points.
        In such cases, you are better off calling the
        `populate_mock` method and then calling the
        `~halotools.mock_observables.tpcf` after placing a cut on the
        ``galaxy_table``, as demonstrated in :ref:`galaxy_catalog_analysis_tutorial2`.
        """
        if summary_statistic == 'mean':
            summary_func = np.mean
        else:
            summary_func = np.median

        halocat_kwargs = {}
        if 'simname' in kwargs:
            halocat_kwargs['simname'] = kwargs['simname']
        if 'redshift' in kwargs:
            halocat_kwargs['redshift'] = kwargs['redshift']
        elif hasattr(self, 'redshift'):
            halocat_kwargs['redshift'] = self.redshift
        if 'halo_finder' in kwargs:
            halocat_kwargs['halo_finder'] = kwargs['halo_finder']

        try:
            assert kwargs['simname'] == 'fake'
            use_fake_sim = True
        except (AssertionError, KeyError):
            use_fake_sim = False

        if use_fake_sim is True:
            halocat = FakeSim(**halocat_kwargs)
        else:
            halocat = CachedHaloCatalog(preload_halo_table=True, **halocat_kwargs)

        if 'rbins' in kwargs:
            rbins = kwargs['rbins']
        else:
            rbins = model_defaults.default_rbins

        if 'include_crosscorr' in list(kwargs.keys()):
            include_crosscorr = kwargs['include_crosscorr']
        else:
            include_crosscorr = False

        if include_crosscorr is True:

            xi_coll = np.zeros(
                (len(rbins)-1)*num_iterations*3).reshape(3, num_iterations, len(rbins)-1)

            for i in range(num_iterations):
                self.populate_mock(halocat=halocat)
                rbin_centers, xi_coll[0, i, :], xi_coll[1, i, :], xi_coll[2, i, :] = (
                    self.mock.compute_galaxy_clustering(**kwargs)
                    )
            xi_11 = summary_func(xi_coll[0, :], axis=0)
            xi_12 = summary_func(xi_coll[1, :], axis=0)
            xi_22 = summary_func(xi_coll[2, :], axis=0)
            return rbin_centers, xi_11, xi_12, xi_22
        else:

            xi_coll = np.zeros(
                (len(rbins)-1)*num_iterations).reshape(num_iterations, len(rbins)-1)

            for i in range(num_iterations):
                self.populate_mock(halocat=halocat)
                rbin_centers, xi_coll[i, :] = self.mock.compute_galaxy_clustering(**kwargs)
            xi = summary_func(xi_coll, axis=0)
            return rbin_centers, xi

    def compute_average_galaxy_matter_cross_clustering(self, num_iterations=5,
            summary_statistic='median', **kwargs):
        r"""
        Method repeatedly populates a simulation with a mock galaxy catalog,
        computes the galaxy-matter cross-correlation
        signal of each Monte Carlo realization, and returns a summary statistic of the clustering
        such as the median computed from the collection of repeated measurements.

        The `compute_average_galaxy_matter_cross_clustering` is simply a convenience function,
        and is not intended for use in performance-critical applications such as MCMCs.
        In an MCMC, there is no need to repeatedly populate the same snapshot with the same
        set of model parameters; the primary purpose for this repetition is for smoothing out
        numerical noise when making plots and doing exploratory work.
        If you wish to use the 3d cross-correlation function in a performance-critical application,
        see :ref:`galaxy_catalog_analysis_tutorial2` for a demonstration of how to
        call the `~halotools.mock_observables.tpcf` function once,
        directly on the mock galaxy catalog, and then refer to the
        `~halotools.mock_observables.tpcf` docstring for how to use the cross-correlation feature.

        Parameters
        ----------
        num_iterations : int, optional
            Number of Monte Carlo realizations to use to estimate the clustering signal.
            Default is 5.

        summary_statistic : string, optional
            String specifying the method used to estimate the clustering signal from the
            collection of Monte Carlo realizations. Options are ``median`` and ``mean``.
            Default is ``median``.

        simname : string, optional
            Nickname of the simulation into which mock galaxies will be populated.
            Currently supported simulations are
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).
            Default is set in `~halotools.sim_manager.sim_defaults`.

        halo_finder : string, optional
            Nickname of the halo-finder of the halocat into which mock galaxies
            will be populated, e.g., `rockstar` or `bdm`.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        redshift : float, optional
            Redshift of the desired halocat into which mock galaxies will be populated.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        variable_galaxy_mask : scalar, optional
            Any value used to construct a mask to select a sub-population
            of mock galaxies. See examples below.

        mask_function : array, optional
            Function object returning a masking array when operating on the galaxy_table.
            More flexible than the simpler ``variable_galaxy_mask`` option because ``mask_function``
            allows for the possibility of multiple simultaneous cuts. See examples below.

        include_complement : bool, optional
            Only for simultaneous use with a ``variable_galaxy_mask``-determined mask.
            If ``include_complement`` is set to False (the default option), method will return
            the cross-correlation function between a random downsampling of dark matter particles
            and the subsample of galaxies determined by
            the input ``variable_galaxy_mask``. If ``include_complement`` is True,
            method will also return the cross-correlation between the dark matter particles
            and the complementary subsample. See examples below.

        rbins : array, optional
            Bins in which the correlation function will be calculated.
            Default is set in `~halotools.empirical_models.model_defaults` module.

        Examples
        ---------
        The simplest use-case of the `compute_average_galaxy_matter_cross_clustering` function
        is just to call the function with no arguments. This will generate a sequence
        of Monte Carlo realizations of your model into the default halocat,
        calculate the cross-correlation function between dark matter
        and all galaxies in your mock, and return the median
        clustering strength in each radial bin:

        >>> model = Leauthaud11() # doctest: +SKIP
        >>> r, clustering = model.compute_average_galaxy_matter_cross_clustering() # doctest: +SKIP

        To control how which simulation is used, you use the same syntax you use to load
        a `~halotools.sim_manager.CachedHaloCatalog` into memory from your cache directory:

        >>> r, clustering = model.compute_average_galaxy_matter_cross_clustering(simname = 'multidark', redshift=1) # doctest: +SKIP

        You can control the number of mock catalogs that are generated via:

        >>> r, clustering = model.compute_average_galaxy_matter_cross_clustering(num_iterations = 10) # doctest: +SKIP

        You may wish to focus on the clustering signal for a specific subpopulation. To do this,
        you have two options. First, you can use the ``variable_galaxy_mask`` mechanism:

        >>> r, clustering = model.compute_average_galaxy_matter_cross_clustering(gal_type = 'centrals') # doctest: +SKIP

        With the ``variable_galaxy_mask`` mechanism, you are free to use any column of your galaxy_table
        as a keyword argument. If you couple this function call with the ``include_complement``
        keyword argument, the function will also return the correlation function of the complementary subset.

        >>> r, cen_clustering, sat_clustering = model.compute_average_galaxy_matter_cross_clustering(gal_type = 'centrals', include_complement = True) # doctest: +SKIP

        Returns
        --------
        rbin_centers : array
            Midpoint of the bins used in the correlation function calculation

        correlation_func : array
            If not using any mask (the default option), method returns the
            correlation function of the full mock galaxy catalog.

            If using a mask, and if ``include_crosscorr`` is False (the default option),
            method returns the correlation function of the subsample of galaxies determined by
            the input mask.

            If using a mask, and if ``include_crosscorr`` is True,
            method will return the auto-correlation of the subsample,
            the cross-correlation of the subsample and the complementary subsample,
            and the the auto-correlation of the complementary subsample, in that order.
            See the example below.

        Notes
        -----
        The `compute_average_galaxy_matter_cross_clustering` method bound to
        mock instances is just a convenience wrapper
        around the `~halotools.mock_observables.tpcf` function. If you wish for greater
        control over how your galaxy clustering signal is estimated,
        see the `~halotools.mock_observables.tpcf` documentation.

        Note that there can be no guarantees that the
        `compute_average_galaxy_matter_cross_clustering` method bound to your model
        will terminate in a reasonable amount of time. For example,
        if you use a subhalo-based model that populates *every* subhalo
        in the catalog with a mock galaxy, then calling
        `compute_average_galaxy_matter_cross_clustering` on this model will attempt
        to compute a correlation function on hundreds of millions of points.
        In such cases, you are better off calling the
        `populate_mock` method and then calling the
        `~halotools.mock_observables.tpcf` after placing a cut on the
        ``galaxy_table``, as demonstrated in :ref:`galaxy_catalog_analysis_tutorial3`.
        The only difference between this use-case and the one demonstrated in
        the tutorial is that here you will use the `~halotools.mock_observables.tpcf`
        to calculate the cross-correlation between dark matter particles and galaxies,
        rather than calling the `~halotools.mock_observables.mean_delta_sigma` function.

        """
        if summary_statistic == 'mean':
            summary_func = np.mean
        else:
            summary_func = np.median

        halocat_kwargs = {}
        if 'simname' in kwargs:
            halocat_kwargs['simname'] = kwargs['simname']
        if 'redshift' in kwargs:
            halocat_kwargs['redshift'] = kwargs['redshift']
        elif hasattr(self, 'redshift'):
            halocat_kwargs['redshift'] = self.redshift
        if 'halo_finder' in kwargs:
            halocat_kwargs['halo_finder'] = kwargs['halo_finder']

        try:
            assert kwargs['simname'] == 'fake'
            use_fake_sim = True
        except (AssertionError, KeyError):
            use_fake_sim = False

        if use_fake_sim is True:
            halocat = FakeSim(num_ptcl=int(1e5), **halocat_kwargs)
        else:
            halocat = CachedHaloCatalog(preload_halo_table=True, **halocat_kwargs)

        if 'rbins' in kwargs:
            rbins = kwargs['rbins']
        else:
            rbins = model_defaults.default_rbins

        if 'include_complement' in list(kwargs.keys()):
            include_complement = kwargs['include_complement']
        else:
            include_complement = False

        if include_complement is True:

            xi_coll = np.zeros(
                (len(rbins)-1)*num_iterations*2).reshape(2, num_iterations, len(rbins)-1)

            for i in range(num_iterations):
                self.populate_mock(halocat=halocat)
                rbin_centers, xi_coll[0, i, :], xi_coll[1, i, :] = (
                    self.mock.compute_galaxy_matter_cross_clustering(**kwargs)
                    )
            xi_11 = summary_func(xi_coll[0, :], axis=0)
            xi_22 = summary_func(xi_coll[1, :], axis=0)
            return rbin_centers, xi_11, xi_22
        else:

            xi_coll = np.zeros(
                (len(rbins)-1)*num_iterations).reshape(num_iterations, len(rbins)-1)

            for i in range(num_iterations):
                self.populate_mock(halocat=halocat)
                rbin_centers, xi_coll[i, :] = self.mock.compute_galaxy_matter_cross_clustering(**kwargs)
            xi = summary_func(xi_coll, axis=0)
            return rbin_centers, xi