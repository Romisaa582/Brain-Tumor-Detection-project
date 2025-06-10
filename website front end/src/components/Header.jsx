import React from "react";
import { assets } from "../assets/assets";

const Header = () => {
  return (
    <div className="relative bg-primary rounded-lg overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-white/20 via-transparent to-transparent z-0 pointer-events-none" />

      <div className="relative z-10 flex flex-col md:flex-row md:justify-between flex-wrap px-6 md:px-10 lg:px-20">
        <div className="md:w-1/2 flex flex-col items-start justify-center gap-4 py-10 m-auto md:py-[10vw] md:mb-[-30px]">
          <p className="text-3xl md:text-4xl lg:text-5xl text-white font-semibold leading-tight">
            Detect Cancer <br />
            Early with AI Technology
          </p>

          <div className="flex flex-col md:flex-row items-center gap-3 text-white text-sm font-light">
            <img className="w-28" src={assets.group_profiles} alt="Group Profiles" />
            <p>
              Advanced technology for early cancer detection through,
              <br className="hidden sm:block" />
              x-rays.
            </p>
          </div>

          <a
            href="#speciality"
            className="flex items-center gap-2 bg-white px-8 py-3 rounded-full text-sm m-auto md:m-0 hover:scale-105 transition-all duration-300"
          >
            Book Appointment
            <img className="w-3" src={assets.arrow_icon} alt="Arrow Icon" />
          </a>
        </div>

        {/* <div className="md:w-1/2 md:h-[500px] relative">
          <img
            className="ml-auto w-1/2 h-full md:absolute right-0 top-0 rounded-lg object-cover"
            src={assets.header_img}
            alt="Header"
          />
        </div> */}

<div className="md:w-1/2 md:h-[540px] relative pt-6"> {/* ارتفاع أقل و padding أخف */}
  <img
    className="ml-auto w-[65%] h-full md:absolute right-0 top-0 rounded-lg object-cover"
    src={assets.header_img}
    alt="Header"
  />
</div>

      </div>
    </div>
  );
};

export default Header;

