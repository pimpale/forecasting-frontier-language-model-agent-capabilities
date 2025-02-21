import ApolloLogo from '../assets/images/apollo_logo.png';
import MatsLogo from '../assets/images/mats_logo.png';

const Header = () => {
  return (
    <header className="navbar static-top bg-white shadow-sm">
      <div className="container d-flex justify-content-evenly">
        <a href="https://www.apolloresearch.ai/">
          <img
            src={ApolloLogo}
            alt="Apollo Logo"
            style={{ height: '40px' }}
          />
        </a>
        <a href="https://www.matsprogram.org/">
          <img
            src={MatsLogo}
            alt="MATS Logo"
            style={{ height: '40px' }}
          />
        </a>
      </div>
    </header>
  );
};

export default Header;